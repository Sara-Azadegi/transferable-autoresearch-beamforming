"""
train.py -- ONLY mutable research source file for Baby Percentile Beamforming.

Initial seed (C3.1-C3.19, frozen before the campaign):
  * centralized full-CSI MLP,
  * input = asinh(Re H/sigma), asinh(Im H/sigma), unit-norm baseline beams
    (Re, Im), Kq/8 (641 real scalars),
  * 641 -> 512 -> 512 -> 128 with ReLU; last layer zero-initialized,
  * output V = V_base + sqrt(Pmax/4) * correction, V_base = equal-power dominant-SVD
    beams; per-BS rescale only if a BS exceeds Pmax (untrained model = baseline),
  * direct raw exact-SLqP training loss,
  * Adam, lr=1e-3, 2000 steps, batch=128,
  * fresh Baby-v1.1 channels every step, Kq ~ Uniform{1,...,8},
  * CPU, float32 network, float64/complex128 training physics,
  * full autograd through the analytical MMSE torch.linalg.solve,
   * no scheduler, dropout, weight decay, or gradient clipping,
  * only the final step-2000 model is evaluated/saved.

The immutable prepare.py is the authoritative judge. The differentiable physics
below exists ONLY to provide the seed training objective and may be redesigned by
the research agent later, provided the immutable evaluator contract is respected.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(20)

from prepare import (
    B,
    BANDWIDTH_HZ,
    K_TOTAL,
    K_USERS,
    M_TX,
    N_RX,
    NOISE_POWER_W,
    P_MAX_W,
    SQRT_P_MAX_W,
    SEED_TRAIN,
    TORCH_SEED,
    TrainingChannelStream,
    evaluate_selection,
)

# =============================================================================
# Frozen initial seed-training recipe
# =============================================================================
DEVICE = torch.device("cpu")
TRAIN_REAL_DTYPE = torch.float32
TRAIN_COMPLEX_DTYPE = torch.complex64
PHYS_COMPLEX_DTYPE = torch.complex128   # training physics only
PHYS_REAL_DTYPE = torch.float64

STEPS = 4000                    # Exp 4: max budget (was 2000)
BATCH = 256                     # Exp 5: was 128
LR = 2e-3                        # Exp 17: was 1e-3
NORM_EMA = 0.99                 # Exp 2: per-Kq baseline normalizer EMA decay
FAMILY = "interference_gnn"
TRAIN_TIME_LIMIT_S = 600.0
# Calibration-only seed overrides. Not set during the campaign.
RUN_TORCH_SEED = int(os.environ.get("CALIB_TORCH_SEED", TORCH_SEED))
RUN_TRAIN_SEED = int(os.environ.get("CALIB_TRAIN_SEED", SEED_TRAIN))

INPUT_DIM = 2 * K_USERS * B * B * N_RX * M_TX + 2 * K_USERS * B * M_TX + 1   # 512 + 128 + 1 = 641
# Exp 3: + 16 log effective gains + 8 SLNR-init spectral efficiencies.
N_LINK_FEAT = K_TOTAL * B + K_TOTAL
INPUT_DIM = INPUT_DIM + N_LINK_FEAT                    # 665
# Exp 1: structured output = 16 log-regularization weights (one per receiving
# user x transmitting BS) + 8 per-BS power logits.
N_REG_OUT = K_TOTAL * B
N_POW_OUT = K_TOTAL
OUTPUT_DIM = N_REG_OUT + N_POW_OUT                    # 24 real outputs

STREAM_AMP = math.sqrt(P_MAX_W / K_USERS)   # sqrt(Pmax/4)
REG_Q0_W = P_MAX_W / K_USERS                 # initial "uplink" weight q = Pmax/4 (SLNR init)
LOG_Q_CLAMP = 25.0


def abs2(x: torch.Tensor) -> torch.Tensor:
    """|x|^2 for complex x as Re^2 + Im^2 (faster than abs().square())."""
    return x.real.square() + x.imag.square()


def cnorm(x: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the last dim, keepdim (replaces vector_norm)."""
    return torch.sqrt(abs2(x).sum(dim=-1, keepdim=True))


def dominant_left_singular(A: torch.Tensor) -> torch.Tensor:
    """Closed-form dominant left singular vector of [...,2,M] matrices via the
    principal eigenvector of the 2x2 Hermitian A A^H (replaces batched SVD;
    identical up to an irrelevant global phase)."""
    R = A @ A.conj().transpose(-1, -2)                                 # [...,2,2]
    a = R[..., 0, 0].real
    d = R[..., 1, 1].real
    b = R[..., 0, 1]
    lam = 0.5 * (a + d) + torch.sqrt((0.5 * (a - d)).square() + abs2(b))
    first = a >= d
    u0 = torch.where(first, (lam - d).to(b.dtype), b)
    u1 = torch.where(first, b.conj(), (lam - a).to(b.dtype))
    u = torch.stack((u0, u1), dim=-1)
    return u / cnorm(u).clamp_min(1e-300)


def baseline_beams(Hc: torch.Tensor) -> torch.Tensor:
    """Equal-power dominant-SVD beams (same rule as the evaluator's floor)."""
    direct = torch.stack([Hc[:, :, b, b] for b in range(B)], dim=2)   # [batch,K,B,N,M]
    u = dominant_left_singular(direct)                                 # [batch,K,B,N]
    v = torch.einsum("tkbn,tkbnm->tkbm", u.conj(), direct).conj()      # H^H u
    return STREAM_AMP * v / cnorm(v)


def effective_channels(H128: torch.Tensor) -> torch.Tensor:
    """Effective MISO rows G[t,j,c,b,:] = u_{j,c}^H H[t,j,c,b], u = dominant left
    singular vector of the user's direct channel. No gradient is needed here."""
    direct = torch.stack([H128[:, :, c, c] for c in range(B)], dim=2)  # [T,K,B,N,M]
    u = dominant_left_singular(direct)                                 # [T,K,B,N]
    return torch.einsum("tjcn,tjcbnm->tjcbm", u.conj(), H128)          # [T,K,B,B,M]


def structured_beams(G: torch.Tensor, log_q: torch.Tensor, pow_logits: torch.Tensor) -> torch.Tensor:
    """Regularized-ZF / SLNR-structure beams with learned weights and power split.

    G         : complex128 [T,K,C,B,M] effective row from BS b to user (k,c)
    log_q     : real [T,K,C,B] log of q_{k,c,b}/REG_Q0_W (weight of user (k,c) at BS b)
    pow_logits: real [T,K,B] per-BS softmax power logits
    Returns V complex128 [T,K,B,M] with sum_k ||v_{k,b}||^2 = Pmax exactly.
    """
    T = G.shape[0]
    w = (REG_Q0_W / NOISE_POWER_W) * torch.exp(torch.clamp(log_q.to(PHYS_REAL_DTYPE), -LOG_Q_CLAMP, LOG_Q_CLAMP))
    Gb = G.permute(0, 3, 1, 2, 4).reshape(T, B, K_TOTAL, M_TX)          # [T,B,(k,c),M]
    wb = w.permute(0, 3, 1, 2).reshape(T, B, K_TOTAL)                   # [T,B,(k,c)]
    eye = torch.eye(M_TX, dtype=PHYS_COMPLEX_DTYPE, device=G.device)
    A = eye + torch.einsum("tbj,tbjm,tbjn->tbmn", wb.to(PHYS_COMPLEX_DTYPE), Gb.conj(), Gb)
    # Right-hand sides: conj of BS b's own users' effective rows.
    rhs = torch.stack([G[:, :, b, b, :] for b in range(B)], dim=1).conj()   # [T,B,K,M]
    D = torch.linalg.solve(A, rhs.transpose(-1, -2)).transpose(-1, -2)    # [T,B,K,M]
    D = D / cnorm(D)
    p = P_MAX_W * torch.softmax(pow_logits.to(PHYS_REAL_DTYPE), dim=1)     # [T,K,B]
    V = D.permute(0, 2, 1, 3) * torch.sqrt(p).unsqueeze(-1)              # [T,K,B,M]
    return V


class StructuredBeamMLP(nn.Module):
    """Exp 1: seed MLP trunk with a structured RZF/SLNR beamforming output head."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, OUTPUT_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _kq_vector(Kq: int | torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(Kq, int):
            kq = torch.full((batch,), Kq, dtype=torch.long, device=device)
        else:
            kq = torch.as_tensor(Kq, dtype=torch.long, device=device)
            if kq.ndim == 0:
                kq = kq.expand(batch)
        if tuple(kq.shape) != (batch,):
            raise ValueError(f"Kq shape {tuple(kq.shape)} != ({batch},)")
        if bool(((kq < 1) | (kq > K_TOTAL)).any()):
            raise ValueError(f"all Kq values must lie in 1..{K_TOTAL}")
        return kq

    def forward(self, H: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
        expected_tail = (K_USERS, B, B, N_RX, M_TX)
        if H.ndim != 6 or tuple(H.shape[1:]) != expected_tail:
            raise ValueError(f"H shape {tuple(H.shape)} is incompatible with Baby-v1.1")
        if not torch.is_complex(H):
            raise ValueError("H must be complex")

        Hc = H.to(device=DEVICE, dtype=TRAIN_COMPLEX_DTYPE)
        batch = Hc.shape[0]
        kq = self._kq_vector(Kq, batch, Hc.device)

        Hs = Hc / math.sqrt(NOISE_POWER_W)
        xr = torch.asinh(Hs.real).reshape(batch, -1)
        xi = torch.asinh(Hs.imag).reshape(batch, -1)
        xk = (kq.to(TRAIN_REAL_DTYPE) / float(K_TOTAL)).unsqueeze(1)

        V0 = baseline_beams(Hc)                                   # baseline beams (input features)
        v0n = V0 / STREAM_AMP                                     # unit-norm beams as features

        # Structured layer in complex128 (ill-conditioned for users near a BS).
        H128 = H.to(device=DEVICE, dtype=PHYS_COMPLEX_DTYPE)
        G = effective_channels(H128)

        # Exp 3: explicit per-user link-quality features (deterministic, no grad).
        with torch.no_grad():
            gain = G.abs().square().sum(dim=-1)                           # [T,k,c,b]
            xg = torch.log10(gain * (REG_Q0_W / NOISE_POWER_W) + 1e-30) / 4.0
            V_init = structured_beams(
                G,
                torch.zeros((batch, K_USERS, B, B), dtype=PHYS_REAL_DTYPE, device=DEVICE),
                torch.zeros((batch, K_USERS, B), dtype=PHYS_REAL_DTYPE, device=DEVICE),
            )
            se_init = training_rates_mbps(H128, V_init) / (BANDWIDTH_HZ / 1e6)   # [T,k,b] bit/s/Hz
            xs = se_init / 4.0
        x = torch.cat((xr, xi,
                       v0n.real.reshape(batch, -1), v0n.imag.reshape(batch, -1),
                       xk,
                       xg.reshape(batch, -1).to(TRAIN_REAL_DTYPE),
                       xs.reshape(batch, -1).to(TRAIN_REAL_DTYPE)), dim=1)

        out = self.net(x)
        log_q = out[:, :N_REG_OUT].reshape(batch, K_USERS, B, B)          # [T,k,c,b]
        pow_logits = out[:, N_REG_OUT:].reshape(batch, K_USERS, B)        # [T,k,b]

        # Zero-init head -> equal-power SLNR beams with q = Pmax/4 for every user.
        # Softmax power split makes every BS use exactly Pmax (feasible by construction).
        return structured_beams(G, log_q, pow_logits)


GNN_HIDDEN = 160                # Exp 25: was 128
GNN_LAYERS = 3
GNN_LAYERS2 = 2       # Exp 8: refinement-stage message-passing layers (exp 22: was 1)
N_EXTRA_ALT = 3       # Exp 14: extra fixed transmit/receive alternations after exp 10's
N_ALT1 = 3            # Exp 15: alternations of the stage-1 beamformer before refinement
N_ALT2 = 2            # Exp 15: alternations of the refined beamformer (5 total, as run 14)
NODE_FEAT2 = 4        # Exp 8: stage-1 SE + 3 Kq-relative rank features
NODE_FEAT = 7          # xg own BS, xg other BS, SLNR-init SE, Kq/8, + exp 7 rank feats
EDGE_FEAT = 5          # corr at BS(i), corr at BS(j), same-cell, xg_{i<-BS(j)}, xg_{j<-BS(i)}
EDGE_FEAT2 = EDGE_FEAT + 2   # Exp 20: + post-combining INR j->i and i->j under stage-1 beams


def rank_features(se: torch.Tensor, kq: torch.Tensor) -> torch.Tensor:
    """Kq-relative rank features per user from per-user SE [T,8] (0 = weakest)."""
    rank = torch.argsort(torch.argsort(se, dim=1), dim=1).to(PHYS_REAL_DTYPE)
    kq_f = kq.to(PHYS_REAL_DTYPE).view(-1, 1)
    return torch.stack((rank / (K_TOTAL - 1), (rank < kq_f).to(PHYS_REAL_DTYPE),
                        (kq_f - 1.0 - rank) / K_TOTAL), dim=-1)


class InterferenceGNN(nn.Module):
    """Family interference_gnn (exp 6): permutation-equivariant message passing
    over the 8 user nodes, same structured RZF/SLNR + softmax-power output layer.

    Node i = (k, c) flattened as i = k*B + c. Messages are aggregated separately
    over same-cell and other-cell neighbours. Output per node: power logit,
    log q at own BS, log q at other BS (zero-init head -> equal-power SLNR).
    """

    def __init__(self) -> None:
        super().__init__()
        Hd = GNN_HIDDEN
        self.enc = nn.Sequential(nn.Linear(NODE_FEAT, Hd), nn.ReLU(), nn.Linear(Hd, Hd))
        self.msg_i = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(GNN_LAYERS)])
        self.msg_j = nn.ModuleList([nn.Linear(Hd, Hd, bias=False) for _ in range(GNN_LAYERS)])
        self.msg_e = nn.ModuleList([nn.Linear(EDGE_FEAT, Hd, bias=False) for _ in range(GNN_LAYERS)])
        self.msg_out = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(GNN_LAYERS)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * Hd, Hd), nn.ReLU(), nn.Linear(Hd, Hd))
            for _ in range(GNN_LAYERS)
        ])
        self.head = nn.Linear(Hd, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Exp 8: refinement stage (created after stage 1 so stage-1 init is unchanged).
        self.enc2 = nn.Sequential(nn.Linear(Hd + NODE_FEAT2, Hd), nn.ReLU(), nn.Linear(Hd, Hd))
        self.msg2_i = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(GNN_LAYERS2)])
        self.msg2_j = nn.ModuleList([nn.Linear(Hd, Hd, bias=False) for _ in range(GNN_LAYERS2)])
        self.msg2_e = nn.ModuleList([nn.Linear(EDGE_FEAT2, Hd, bias=False) for _ in range(GNN_LAYERS2)])
        self.msg2_out = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(GNN_LAYERS2)])
        self.upd2 = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * Hd, Hd), nn.ReLU(), nn.Linear(Hd, Hd))
            for _ in range(GNN_LAYERS2)
        ])
        self.head2 = nn.Linear(Hd, 3)
        nn.init.zeros_(self.head2.weight)
        nn.init.zeros_(self.head2.bias)

        cell = torch.arange(K_TOTAL) % B
        same = (cell[:, None] == cell[None, :]).to(TRAIN_REAL_DTYPE)
        offdiag = 1.0 - torch.eye(K_TOTAL, dtype=TRAIN_REAL_DTYPE)
        self.register_buffer("mask_same", same * offdiag, persistent=False)     # 3 per row
        self.register_buffer("mask_other", 1.0 - same, persistent=False)        # 4 per row

    _kq_vector = staticmethod(StructuredBeamMLP._kq_vector)

    def _edge_features(self, G: torch.Tensor, xg: torch.Tensor) -> torch.Tensor:
        """Edge features for ordered pair (i, j) from effective rows G [T,k,c,b,M]."""
        batch = G.shape[0]
        Gn = G / cnorm(G)
        Gn = Gn.reshape(batch, K_TOTAL, B, M_TX)                            # [T,i,b,M]
        corr = abs2(torch.einsum("tibm,tjbm->tijb", Gn.conj(), Gn))   # [T,i,j,b]
        cell = torch.arange(K_TOTAL, device=DEVICE) % B
        ci = cell.view(1, K_TOTAL, 1, 1).expand(batch, K_TOTAL, K_TOTAL, 1)
        cj = cell.view(1, 1, K_TOTAL, 1).expand(batch, K_TOTAL, K_TOTAL, 1)
        corr_i = corr.gather(-1, ci).squeeze(-1)
        corr_j = corr.gather(-1, cj).squeeze(-1)
        xg_n = xg.reshape(batch, K_TOTAL, B)                                 # [T,i,b]
        xg_i_from_j = xg_n.unsqueeze(2).expand(batch, K_TOTAL, K_TOTAL, B).gather(-1, cj).squeeze(-1)
        xg_j_from_i = xg_n.unsqueeze(1).expand(batch, K_TOTAL, K_TOTAL, B).gather(-1, ci).squeeze(-1)
        same = self.mask_same.to(PHYS_REAL_DTYPE).unsqueeze(0).expand(batch, K_TOTAL, K_TOTAL)
        return torch.stack((corr_i, corr_j, same, xg_i_from_j, xg_j_from_i), dim=-1)

    def forward(self, H: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
        expected_tail = (K_USERS, B, B, N_RX, M_TX)
        if H.ndim != 6 or tuple(H.shape[1:]) != expected_tail:
            raise ValueError(f"H shape {tuple(H.shape)} is incompatible with Baby-v1.1")
        if not torch.is_complex(H):
            raise ValueError("H must be complex")
        batch = H.shape[0]
        kq = self._kq_vector(Kq, batch, DEVICE)

        H128 = H.to(device=DEVICE, dtype=PHYS_COMPLEX_DTYPE)
        G = effective_channels(H128)                                        # [T,k,c,b,M]

        def log_gains(Geff):
            gain = abs2(Geff).sum(dim=-1)                          # [T,k,c,b]
            return torch.log10(gain * (REG_Q0_W / NOISE_POWER_W) + 1e-30) / 4.0

        with torch.no_grad():
            xg = log_gains(G)
            V_init = structured_beams(
                G,
                torch.zeros((batch, K_USERS, B, B), dtype=PHYS_REAL_DTYPE, device=DEVICE),
                torch.zeros((batch, K_USERS, B), dtype=PHYS_REAL_DTYPE, device=DEVICE),
            )
            xs = training_rates_mbps(H128, V_init) / (BANDWIDTH_HZ / 1e6) / 4.0  # [T,k,c]

            # Node features, nodes i = k*B + c.
            idx_c = torch.arange(B, device=DEVICE)
            xg_own = xg[:, :, idx_c, idx_c]                                 # [T,k,c]
            xg_oth = xg[:, :, idx_c, 1 - idx_c]
            xk = (kq.to(PHYS_REAL_DTYPE) / float(K_TOTAL)).view(batch, 1, 1).expand(batch, K_USERS, B)
            # Exp 7: Kq-relative rank of each user's SLNR-init SE (0 = weakest).
            rank_feat = rank_features(xs.reshape(batch, K_TOTAL), kq)               # [T,8,3]
            node = torch.cat((torch.stack((xg_own, xg_oth, xs, xk), dim=-1).reshape(batch, K_TOTAL, 4),
                              rank_feat), dim=-1)
            edge = self._edge_features(G, xg)

        h = self.enc(node.to(TRAIN_REAL_DTYPE))                              # [T,8,H]
        n_same = self.mask_same.sum(dim=1).view(1, K_TOTAL, 1)
        n_oth = self.mask_other.sum(dim=1).view(1, K_TOTAL, 1)
        w_same = (self.mask_same / self.mask_same.sum(dim=1, keepdim=True)).view(1, K_TOTAL, K_TOTAL, 1)
        w_oth = (self.mask_other / self.mask_other.sum(dim=1, keepdim=True)).view(1, K_TOTAL, K_TOTAL, 1)
        def mp(h, edge, mi, mj, me, mo, up):
            edge = edge.to(TRAIN_REAL_DTYPE)
            for l in range(len(mi)):
                pre = mi[l](h).unsqueeze(2) + mj[l](h).unsqueeze(1) + me[l](edge)   # [T,i,j,H]
                r = torch.relu(pre)
                # The linear msg_out commutes with the masked mean, so it is applied
                # after aggregation (identical function, ~8x fewer FLOPs).
                # Masked means as broadcast multiply-sum (avoids einsum permute copies).
                agg_s = mo[l]((r * w_same).sum(dim=2))
                agg_o = mo[l]((r * w_oth).sum(dim=2))
                h = h + up[l](torch.cat((h, agg_s, agg_o), dim=-1))
            return h

        def to_beams(out, Geff):
            out = out.reshape(batch, K_USERS, B, 3)                          # node (k,c)
            pow_logits = out[..., 0]                                          # [T,k,b]
            # log_q[t,k,c,b]: channel 1 at the user's own BS (b == c), channel 2 otherwise.
            log_q = torch.stack(
                [torch.stack([out[:, :, c, 1] if b == c else out[:, :, c, 2] for b in range(B)], dim=-1)
                 for c in range(B)], dim=2)
            return structured_beams(Geff, log_q, pow_logits)

        def alternate(out, Geff, n):
            """n fixed transmit/receive alternations with fixed weights (no grad)."""
            for _ in range(n):
                _, u = training_rates_and_combiners(H128, to_beams(out, Geff))
                u = u / cnorm(u)
                Geff = torch.einsum("tjcn,tjcbnm->tjcbm", u.conj(), H128)
            return Geff

        h = mp(h, edge, self.msg_i, self.msg_j, self.msg_e, self.msg_out, self.upd)
        out1 = self.head(h)                                                   # [T,8,3]

        # Exp 15: alternate the STAGE-1 beamformer first (N_ALT1 fixed steps), then let
        # the refinement stage observe that converged operating point: per-user SE/rank
        # under the alternated stage-1 beams and edge features of the alternated geometry.
        with torch.no_grad():
            GA = alternate(out1.detach(), G, N_ALT1)
            V1A = to_beams(out1.detach(), GA)
            r1, u1A = training_rates_and_combiners(H128, V1A)
            # Exp 20: post-combining interference-to-noise ratio from stream j at user i.
            S1 = torch.einsum("tkbcnm,tjcm->tkbjcn", H128, V1A)
            P1 = abs2(torch.einsum("tkbn,tkbjcn->tkbjc", u1A.conj(), S1))
            P1 = P1 / (NOISE_POWER_W * abs2(u1A).sum(dim=-1))[..., None, None]
            inr = torch.log10(P1.reshape(batch, K_TOTAL, K_TOTAL) + 1e-6) / 4.0      # [T,i,j]
            se1 = (r1 / (BANDWIDTH_HZ / 1e6) / 4.0).reshape(batch, K_TOTAL)
            f2 = torch.cat((se1.unsqueeze(-1), rank_features(se1, kq)), dim=-1).to(TRAIN_REAL_DTYPE)
            edge2 = torch.cat((self._edge_features(GA, log_gains(GA)),
                               inr.unsqueeze(-1), inr.transpose(1, 2).unsqueeze(-1)), dim=-1)
        h = h + self.enc2(torch.cat((h, f2), dim=-1))
        h = mp(h, edge2, self.msg2_i, self.msg2_j, self.msg2_e, self.msg2_out, self.upd2)
        out2 = out1 + self.head2(h)
        with torch.no_grad():
            GF = alternate(out2.detach(), GA, N_ALT2)
        return to_beams(out2, GF)


# =============================================================================
# Differentiable TRAINING copy of Baby physics (not the authoritative judge)
# =============================================================================
def training_rates_mbps(H: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Differentiable float64/complex128 MMSE/SINR/rate calculation for seed training."""
    return training_rates_and_combiners(H, V)[0]


def training_rates_and_combiners(H: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rates [T,K,B] (Mbps) and MMSE combiners u [T,K,B,N] (exp 9 helper).

    Vectorized over the flattened (k,b) receivers / (j,c) streams via diagonal
    extraction (identical math to the former per-(k,b) loops; exp 23/24)."""
    H = H.to(device=DEVICE, dtype=PHYS_COMPLEX_DTYPE)
    V = V.to(device=DEVICE, dtype=PHYS_COMPLEX_DTYPE)
    batch = H.shape[0]

    # S[t,i,j,n] = H[t,k,b,c] @ V[t,j',c], receiver i = k*B + b, stream j = j'*B + c
    S = torch.einsum("tkbcnm,tjcm->tkbjcn", H, V).reshape(batch, K_TOTAL, K_TOTAL, N_RX)
    desired = S.diagonal(dim1=1, dim2=2).transpose(1, 2)                 # [T,8,N]

    eye = torch.eye(N_RX, dtype=PHYS_COMPLEX_DTYPE, device=DEVICE).view(1, 1, N_RX, N_RX)
    Ryy = torch.einsum("tijn,tijm->tinm", S, S.conj()) + NOISE_POWER_W * eye

    # C3.18: full backpropagation through the analytical MMSE linear solve.
    u = torch.linalg.solve(Ryy, desired.unsqueeze(-1)).squeeze(-1)       # [T,8,N]

    proj = torch.einsum("tin,tijn->tij", u.conj(), S)
    stream_power = abs2(proj)                                            # [T,8,8]
    signal = stream_power.diagonal(dim1=1, dim2=2)                       # [T,8]
    interference = stream_power.sum(dim=-1) - signal

    noise_out = NOISE_POWER_W * abs2(u).sum(dim=-1)
    sinr = signal / (interference + noise_out)
    rates = torch.log2(1.0 + sinr) * (BANDWIDTH_HZ / 1e6)
    return rates.reshape(batch, K_USERS, B), u.reshape(batch, K_USERS, B, N_RX)


def training_slqp(rates_mbps: torch.Tensor, Kq: torch.Tensor) -> torch.Tensor:
    """C3.13 exact hard SLqP using sort + cumulative sum for mixed-Kq batches."""
    flat = rates_mbps.reshape(rates_mbps.shape[0], K_TOTAL)
    sorted_rates = torch.sort(flat, dim=-1).values
    prefix = torch.cumsum(sorted_rates, dim=-1)
    return prefix.gather(1, (Kq - 1).unsqueeze(1)).squeeze(1)


# =============================================================================
# One fixed-budget seed experiment
# =============================================================================
def train_once() -> Tuple[nn.Module, dict]:
    # C2.31: same learning randomness every complete experiment.
    torch.manual_seed(RUN_TORCH_SEED)

    print(f"FAMILY {FAMILY}")
    print(f"SEEDS torch={RUN_TORCH_SEED} train={RUN_TRAIN_SEED}")

    model = InterferenceGNN().to(device=DEVICE, dtype=TRAIN_REAL_DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # no weight decay
    # Exp 4: cosine LR decay from LR to 0 over STEPS (one scheduler step per update).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=0.0)
    channels = TrainingChannelStream(seed=RUN_TRAIN_SEED)                        # resets SEED_TRAIN=789

    # Separate deterministic generator for Kq task draws. It is seeded from the
    # already-frozen TORCH_SEED so identical train.py runs reproduce the same task
    # sequence without perturbing model initialization.
    kq_gen = torch.Generator(device="cpu").manual_seed(RUN_TORCH_SEED)

    model.train()
    train_t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        H = channels.sample(BATCH, dtype=torch.complex128).to(DEVICE)
        Kq = torch.randint(1, K_TOTAL + 1, (BATCH,), generator=kq_gen, dtype=torch.long)

        # Exp 2: per-Kq normalizer = EMA of the baseline's batch-mean SLqP at each
        # Kq on the same training channels (mirrors the per-Kq score ratio).
        with torch.no_grad():
            base_rates = training_rates_mbps(H, baseline_beams(H))
            base_prefix = torch.cumsum(torch.sort(base_rates.reshape(BATCH, K_TOTAL), dim=-1).values, dim=-1)
            batch_base = base_prefix.mean(dim=0)                    # [8]
            base_ema = batch_base if step == 1 else NORM_EMA * base_ema + (1.0 - NORM_EMA) * batch_base

        V = model(H, Kq)
        rates = training_rates_mbps(H, V)
        slqp = training_slqp(rates, Kq)
        loss = -(slqp / base_ema[Kq - 1]).mean()                 # score-aligned normalized SLqP

        if not torch.isfinite(loss):
            raise RuntimeError(f"training failed: non-finite loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()                                          # no clipping
        optimizer.step()
        scheduler.step()

        train_elapsed = time.perf_counter() - train_t0
        if train_elapsed > TRAIN_TIME_LIMIT_S:
            print(f"TRAIN_S {train_elapsed:.6f}")
            raise RuntimeError(
                f"training-time violation: {train_elapsed:.3f}s > {TRAIN_TIME_LIMIT_S:.1f}s"
            )

        if step == 1 or step % 100 == 0 or step == STEPS:
            print(
                f"STEP {step:04d}/{STEPS}  LOSS {loss.item():.6f}  "
                f"MEAN_SLQP_MBPS {slqp.mean().item():.6f}"
            )

    train_s = time.perf_counter() - train_t0
    print(f"TRAIN_S {train_s:.6f}")

    # C3.14: ONLY the final seed checkpoint is judged.
    result = evaluate_selection(model, verbose=True)

    # C3.19: save final candidate, no intermediate held-out checkpoint selection.
    checkpoint = {
        "model_class": "InterferenceGNN",
        "state_dict": model.state_dict(),
        "heldoutscore": result["HELDOUTSCORE"],
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "torch_seed": RUN_TORCH_SEED,
        "train_seed": RUN_TRAIN_SEED,
        "train_s": train_s,
        "family": FAMILY,
    }
    torch.save(checkpoint, "last_model.pt")
    return model, result


if __name__ == "__main__":
    train_once()


# =============================================================================
# APPEND-ONLY RESEARCH CHANGELOG
# =============================================================================
# Seed / pre-campaign:
#   Centralized full-CSI 513->512->512->128 ReLU MLP; smooth joint per-BS
#   feasible complex beamformer map; direct raw exact-SLqP objective; Adam 1e-3; 2000 steps;
#   batch 128 fresh channels; uniform Kq in 1..8; CPU complex64 training;
#   full differentiation through MMSE; no scheduler/regularization/clipping.
#
# Pre-campaign audit fixes (before experiment 1):
#   asinh(H/sigma) input scaling; float64/complex128 training physics;
#   calibration-only seed overrides CALIB_TORCH_SEED / CALIB_TRAIN_SEED.
#   output reparameterized as baseline SVD beams + zero-init learned correction,
#   baseline beams also fed as input (641 features); per-BS rescale replaces smooth map.
#
# During Autoresearch, append every hypothesis/result here. Do not rewrite or
# delete prior entries.
#
# -----------------------------------------------------------------------------
# Experiment 1 -- PRE-REGISTRATION (family centralized_mlp)
#   Hypothesis: the seed (run 0, 0.998952) fails because its output space (64
#     free complex beam coefficients + per-BS rescale) is too unstructured to
#     learn from 256k samples. Replacing it with an analytical differentiable
#     RZF/SLNR structure on effective MISO channels
#       v_{k,b} ∝ (I + sum_{(j,c)} w_{j,c,b} h_{j,c<-b}^H h_{j,c<-b})^{-1} h_{k,b<-b}^H,
#       h = u^H H, u = dominant left singular vector of the user's direct channel,
#     with the SAME MLP trunk/inputs/optimizer/budget outputting only 16 log
#     weights w (init q=Pmax/4, i.e. equal-power SLNR) and 8 per-BS softmax
#     power logits (each BS uses exactly Pmax), will clearly beat the baseline.
#   Falsifier: HELDOUTSCORE <= 1.020, OR S_KQ8 <= 1.000.
#   Change: output head 128 -> 24; structured beam layer in complex128.
#     Inputs, trunk 641-512-512, Adam 1e-3, 2000 steps, batch 128, Kq uniform,
#     raw SLqP loss all unchanged.
#
# Experiment 1 -- RESULT: COMMIT (run_id 1, candidate commit 4c6755e)
#   HELDOUTSCORE 1.073354 (champion 0.998952 + delta 0.000289 -> banked).
#   S_KQ1..8 = 0.942 0.947 0.949 0.962 0.990 1.055 1.198 1.543.
#   TRAIN_S 43.8, INFERENCE_S 0.179, 2000 steps.
#   Falsifier NOT triggered (score > 1.020, S_KQ8 = 1.543 > 1.000).
#   Interpretation: the structured head learns, but it trades away the small-Kq
#     tasks (Kq<=5 below baseline) for sum rate. The raw-SLqP loss weights a
#     sample by its absolute Mbps (Kq=8 ~ 90-140 Mbps vs Kq=1 ~ 2 Mbps), so the
#     gradient is dominated by large-Kq samples, while HELDOUTSCORE weights every
#     Kq equally in normalized units. Next: align the loss with the score.
#
# -----------------------------------------------------------------------------
# Experiment 2 -- PRE-REGISTRATION (family centralized_mlp, from champion run 1)
#   Hypothesis: run 1's small-Kq deficit is caused by loss/score misalignment.
#     Dividing each sample's SLqP by a per-Kq normalizer (EMA, decay 0.99, of the
#     baseline beams' batch-mean SLqP at that Kq on the same training channels,
#     no grad) makes the training loss an unbiased surrogate of HELDOUTSCORE, so
#     the small-Kq tasks will rise above baseline and the overall score will rise.
#   Falsifier: S_KQ1 <= 1.000, OR HELDOUTSCORE <= 1.073643 (champion + delta).
#   Change: loss = -mean(slqp / ema_base[Kq]); everything else unchanged.
#
# Experiment 2 -- RESULT: COMMIT (run_id 2, candidate commit 21651de)
#   HELDOUTSCORE 1.122792 (champion 1.073354 + delta -> banked).
#   S_KQ1..8 = 1.023 1.036 1.047 1.041 1.040 1.079 1.208 1.508.
#   TRAIN_S 56.2, INFERENCE_S 0.176.
#   Falsifier NOT triggered (S_KQ1 = 1.023 > 1.000; score > 1.073643).
#   Interpretation: loss alignment fixed the small-Kq regression at a tiny
#     Kq=8 cost. But small-Kq gains are still only +2-5%, and the normalized
#     training loss barely moves from its step-1 value (~1.18 = SLNR init),
#     so the trunk adds little beyond the structure. Suspected cause: from raw
#     asinh(H) entries the MLP cannot easily tell WHICH users are weak, which
#     is what power reallocation for small Kq needs.
#
# -----------------------------------------------------------------------------
# Experiment 3 -- PRE-REGISTRATION (family centralized_mlp, from champion run 2)
#   Hypothesis: the trunk cannot extract per-user link strength from raw
#     asinh(H) entries, and small-Kq power reallocation depends on exactly that
#     information. Appending 24 explicit deterministic link-quality features
#     (16 log10 effective gains ||h_{j,c<-b}||^2 * (Pmax/4)/sigma^2, /4; and 8
#     per-user spectral efficiencies under the zero-init equal-power SLNR
#     beamformer, /4) will let the network reallocate power far better, raising
#     the small-Kq scores.
#   Falsifier: mean(S_KQ1..S_KQ4) <= 1.050 (run 2: 1.037), OR
#     HELDOUTSCORE <= 1.123081 (champion + delta).
#   Change: input 641 -> 665 features; everything else unchanged.
#
# Experiment 3 -- RESULT: COMMIT (run_id 3, candidate commit c342e07)
#   HELDOUTSCORE 1.440398 (champion 1.122792 + delta -> banked).
#   S_KQ1..8 = 1.610 1.471 1.355 1.276 1.234 1.284 1.460 1.834.
#   TRAIN_S 67.1, INFERENCE_S 0.239.
#   Falsifier NOT triggered (mean S_KQ1..4 = 1.428 > 1.050).
#   Interpretation: confirmed; link-quality information was the bottleneck.
#     Gains appear at every Kq (the reg. weights also benefit, Kq=8 1.51->1.83).
#     The normalized training loss was still noisy and not saturated at step
#     2000 (~1.35-1.45), suggesting optimization is now a limiting factor.
#
# -----------------------------------------------------------------------------
# Experiment 4 -- PRE-REGISTRATION (family centralized_mlp, from champion run 3)
#   Hypothesis: the run-3 model is optimization-limited (noisy normalized loss
#     not saturated at 2000 steps with constant LR 1e-3). Doubling to 4000
#     steps (charter max) with cosine LR decay 1e-3 -> 0 will converge further
#     and reduce the end-of-training gradient-noise floor.
#   Falsifier: HELDOUTSCORE <= 1.440687 (champion + delta).
#   Change: STEPS 2000 -> 4000; CosineAnnealingLR(T_max=4000, eta_min=0).
#
# Experiment 4 -- RESULT: COMMIT (run_id 4, candidate commit 5d7a90e)
#   HELDOUTSCORE 1.485212 (champion 1.440398 + delta -> banked).
#   S_KQ1..8 = 1.754 1.539 1.395 1.306 1.259 1.305 1.477 1.847.
#   TRAIN_S 146.6, INFERENCE_S 0.242, 4000 steps.
#   Falsifier NOT triggered. Interpretation: optimization-limited confirmed;
#     gains concentrated at small Kq (Kq=1 +0.144), Kq=8 nearly saturated.
#     Per-batch training loss remains very noisy (batch=128, mixed Kq).
#
# -----------------------------------------------------------------------------
# Experiment 5 -- PRE-REGISTRATION (family centralized_mlp, from champion run 4)
#   Hypothesis: with 4000 cosine-decayed steps the model is now limited by
#     stochastic-gradient noise (mixed-Kq hard-SLqP loss on 128 samples).
#     Doubling the batch to 256 fresh channels per update (2x samples seen,
#     ~290 s projected train time) will lower the noise floor and raise the score.
#   Falsifier: HELDOUTSCORE <= 1.485501 (champion + delta).
#   Change: BATCH 128 -> 256; everything else unchanged.
#
# Experiment 5 -- RESULT: COMMIT (run_id 5, candidate commit 8e57bd5)
#   HELDOUTSCORE 1.503005 (champion 1.485212 + delta -> banked).
#   S_KQ1..8 = 1.790 1.557 1.409 1.322 1.281 1.319 1.486 1.859.
#   TRAIN_S 250.1, INFERENCE_S 0.210, 4000 steps, batch 256.
#   Falsifier NOT triggered. Interpretation: gradient noise mattered, but the
#     gain (+0.018) is much smaller than from features (+0.32): sample/data
#     efficiency of the ordered-slot MLP is the more likely remaining limit.
#
# =============================================================================
# FAMILY 2 OPENED: interference_gnn (branch family/interference_gnn)
#   Structural limitation of centralized_mlp: it reads one fixed ordered
#     665-vector, while the optimal policy is permutation-equivariant over the
#     4 users of each cell (and symmetric across cells). It must learn each
#     per-user rule separately per input slot (sample-inefficient: better
#     information gave +0.32, 2x more data only +0.018), and it has no explicit
#     view of pairwise spatial overlap, which governs interference/regularization.
#   Hypothesis for the new family: weight-shared message passing over user
#     nodes with pairwise interference edge features is equivariant by
#     construction and directly sees the interference graph.
#   Global champion during the grace window: run 5 (1.503005) on master.
# -----------------------------------------------------------------------------
# Experiment 6 -- PRE-REGISTRATION (family interference_gnn, attempt 1/5)
#   Hypothesis: a 3-layer message-passing GNN (hidden 128; nodes = 8 users with
#     own/other-BS log gain, SLNR-init SE, Kq/8; edges = effective-channel
#     correlation at each endpoint's BS, same-cell flag, cross gains; separate
#     same-cell / other-cell mean aggregation; per-node head -> power logit,
#     log q own BS, log q other BS) with the identical structured output layer
#     and training recipe of run 5 beats the global champion despite having no
#     raw-H input.
#   Falsifier: HELDOUTSCORE <= 1.503294 (global champion + delta).
#   Change: model StructuredBeamMLP -> InterferenceGNN (365k params); FAMILY label.
#
# Experiment 6 -- RESULT: COMMIT to family-local champion (run_id 6, candidate d2729cb)
#   HELDOUTSCORE 1.617316 (> global champion 1.503005 + delta; first family
#     candidate -> family-local champion S_family = 1.617316).
#   S_KQ1..8 = 1.907 1.647 1.527 1.460 1.420 1.461 1.604 1.912.
#   TRAIN_S 403.3 (higher than the 270 s smoke projection; ~200 s headroom
#     left), INFERENCE_S 0.238.
#   Falsifier NOT triggered. Interpretation: equivariance + explicit pairwise
#     interference graph is a large win at every Kq, largest at mid Kq (4-6:
#     +0.14..+0.14) where users must be traded off against each other.
#   Status COMMIT here = banked on family/interference_gnn only (grace window);
#     the master global champion stays run 5 until the family window closes.
#
# -----------------------------------------------------------------------------
# Experiment 7 -- PRE-REGISTRATION (family interference_gnn, attempt 2/5,
#   from family-local champion run 6)
#   Hypothesis: SLqP depends on which users fall in the bottom-Kq set, i.e. on
#     each user's rank among all 8 relative to Kq. Mean-aggregation message
#     passing represents sorting poorly and the nodes only see Kq/8 and their
#     own SE. Explicit Kq-relative rank features from the SLNR-init SE
#     (rank/7, 1[rank < Kq], (Kq-1-rank)/8; deterministic, no grad) let each
#     user's power/regularization condition on its set membership, helping
#     mid-Kq tasks most.
#   Falsifier: HELDOUTSCORE <= 1.617605 (family champion + delta), OR
#     mean(S_KQ3..6) <= 1.467 (run 6 value).
#   Change: NODE_FEAT 4 -> 7 (rank features); everything else unchanged.
#
# Experiment 7 -- RESULT: COMMIT to family-local champion (run_id 7, candidate d15b2f1)
#   HELDOUTSCORE 1.619656 (> S_family 1.617316 + delta -> S_family = 1.619656).
#   S_KQ1..8 = 1.907 1.648 1.531 1.464 1.425 1.465 1.605 1.913.
#   TRAIN_S 393.6, INFERENCE_S 0.268.
#   Falsifier NOT triggered (mean S_KQ3..6 = 1.471 > 1.467), but the effect is
#     small and uniform: the GNN already inferred most rank information itself.
#     Kq-set membership is not the main remaining bottleneck.
#
# -----------------------------------------------------------------------------
# Experiment 8 -- PRE-REGISTRATION (family interference_gnn, attempt 3/5,
#   from family-local champion run 7)
#   Hypothesis: node features describe the SLNR INITIALIZATION, but the learned
#     policy moves far from it (e.g. Kq=1 min rate nearly doubles), so the GNN
#     allocates from stale per-user rates. A fixed second refinement stage that
#     observes each user's actual SE and Kq-relative rank under the stage-1
#     beamformer (deterministic, no grad), injects them into the stage-1 hidden
#     states, runs 1 more message-passing layer and adds a zero-init correction
#     to the stage-1 outputs, will correct observable mis-allocations.
#     (Fixed 2 stages, no utility-based selection: inference contract §9 holds.)
#   Falsifier: HELDOUTSCORE <= 1.619945 (family champion + delta).
#   Change: + enc2 / 1 refinement MP layer / zero-init head2 (515k params);
#     rank features factored into rank_features(); stage-1 init unchanged.
#     (2 refinement layers were dropped pre-launch: projected train time too
#     close to the 600 s limit.)
#
# Experiment 8 -- RESULT: COMMIT to family-local champion (run_id 8, candidate 68b1cc7)
#   HELDOUTSCORE 1.621672 (> S_family 1.619656 + delta -> S_family = 1.621672).
#   S_KQ1..8 = 1.908 1.651 1.534 1.467 1.428 1.466 1.606 1.913.
#   TRAIN_S 468.9 (131 s headroom), INFERENCE_S 0.384.
#   Falsifier NOT triggered; gain small (+0.002) and uniform, third such step.
#     Trunk-side information refinements are saturating. The remaining ceiling is
#     more likely the beam STRUCTURE itself: transmit directions are designed for
#     the dominant-singular-vector receive direction, ignoring that each 2-antenna
#     MMSE receiver can suppress interference with its second dimension.
#
# -----------------------------------------------------------------------------
# Experiment 9 -- PRE-REGISTRATION (family interference_gnn, attempt 4/5,
#   from family-local champion run 8)
#   Hypothesis: all transmit directions are designed for the fixed dominant-
#     singular-vector receive direction, whereas each 2-antenna MMSE receiver
#     really suppresses interference with its second dimension. Building the
#     FINAL structured beamformer on receive-aware effective rows g' = u1^H H,
#     u1 = normalized MMSE combiner under the stage-1 beamformer (one fixed
#     transmit/receive alternation, deterministic, no grad, no new parameters),
#     matches the transmit design to the receivers actually used and will raise
#     SLqP, especially at large Kq where interference matters most.
#   Falsifier: HELDOUTSCORE <= 1.621961 (family champion + delta), OR
#     S_KQ8 <= 1.913 (run 8: 1.9128).
#   Change: stage-2 beams use G2 = u1^H H instead of G; training physics gains a
#     training_rates_and_combiners() helper (identical rates).
#
# Experiment 9 -- RESULT: COMMIT to family-local champion (run_id 9, candidate db0ace8)
#   HELDOUTSCORE 1.627933 (> S_family 1.621672 + delta -> S_family = 1.627933).
#   S_KQ1..8 = 1.911 1.653 1.539 1.475 1.436 1.476 1.616 1.919.
#   TRAIN_S 482.5 (118 s headroom), INFERENCE_S 0.350.
#   Falsifier NOT triggered (S_KQ8 = 1.919 > 1.913). Interpretation: the
#     receive-direction mismatch was a real structural ceiling; one alternation
#     gives +0.006 (3x the previous step), largest at Kq 4-7.
#
# -----------------------------------------------------------------------------
# Experiment 10 -- PRE-REGISTRATION (family interference_gnn, attempt 5/5,
#   from family-local champion run 9)
#   Hypothesis: one transmit/receive alternation has not converged: the MMSE
#     combiners under the stage-2 beamformer differ from the stage-1 combiners
#     it was designed for. A second fixed alternation (normalized MMSE combiners
#     u2 under the stage-2 beams, no grad; final beams rebuilt on g'' = u2^H H
#     with the same learned weights; no new parameters) tightens the match.
#   Falsifier: HELDOUTSCORE <= 1.628222 (family champion + delta).
#   Change: final beams = to_beams(out2, G3) with G3 from stage-2 combiners.
#
# Experiment 10 -- RESULT: COMMIT to family-local champion (run_id 10, candidate 8f99ea6)
#   HELDOUTSCORE 1.632548 (> S_family 1.627933 + delta -> S_family = 1.632548).
#   S_KQ1..8 = 1.914 1.656 1.543 1.481 1.442 1.482 1.620 1.921.
#   TRAIN_S 528.3 (only ~72 s headroom left), INFERENCE_S 0.411.
#   Falsifier NOT triggered. Interpretation: alternation not converged after
#     one step; the second adds +0.0046 (diminishing vs +0.0063).
#
# FAMILY interference_gnn GRACE WINDOW CLOSED (5/5 attempts: runs 6-10)
#   Best family-local candidate: run 10, S_family = 1.632548.
#   S_family > S_global (run 5, 1.503005) + delta 0.000289 -> PROMOTED to
#     global champion (merged into master).
#
# -----------------------------------------------------------------------------
# Experiment 11 -- PRE-REGISTRATION (family interference_gnn, depth on global
#   champion run 10, 1.632548)
#   Hypothesis: the hard-sort SLqP gradient (1 for the momentary bottom-Kq
#     users, 0 otherwise) is high-variance and myopic: near-tied users at the
#     membership boundary (typical near a max-min optimum) flip in and out,
#     and the policy is never pushed to equalize them. Soft, mass-preserving
#     membership weights w_i = sigmoid((t* - log r_i)/tau), sum_i w_i = Kq by
#     bisection (no grad; Kq=8 -> 1), loss = -mean(sum_i w_i r_i / c_Kq) with w
#     detached, tau annealed linearly 0.2 -> 0 (ends on the exact hard
#     objective), give near-tied users shared gradient and improve small Kq.
#   Falsifier: HELDOUTSCORE <= 1.632837 (champion + delta), OR
#     mean(S_KQ1..3) <= 1.7046 (run 10 value).
#   Change: training loss only (soft_membership_weights); model unchanged.
#
# Experiment 11 -- RESULT: REVERT (run_id 11, candidate 6c1ffc9)
#   HELDOUTSCORE 1.630020 <= 1.632837 -> reverted to champion run 10 code.
#   S_KQ1..8 = 1.910 1.653 1.539 1.477 1.440 1.480 1.619 1.921.
#   TRAIN_S 527.9, INFERENCE_S 0.401.
#   Falsifier TRIGGERED (mean S_KQ1..3 = 1.7009 < 1.7046). Interpretation: the
#     hard-sort subgradient is not the bottleneck; averaging over the batch
#     already smooths membership flips, and the soft surrogate optimizes a
#     biased (wider-membership) objective for most of training, slightly
#     hurting exactly the small-Kq tasks it targeted.
#
# -----------------------------------------------------------------------------
# Experiment 12 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 10)
#   Hypothesis: training is online (fresh channels, no overfitting), so the
#     final weights carry stochastic-gradient noise from the very noisy mixed-Kq
#     hard-SLqP loss (per-batch swings ~ +/-0.1), only partly removed by cosine
#     decay. Judging a Polyak/EMA average of the weights (decay 0.998, ~500-step
#     horizon, updated every step) instead of the raw final iterate cuts that
#     noise and raises the score. No inference-cost change.
#   Falsifier: HELDOUTSCORE <= 1.632837 (champion + delta).
#   Change: AveragedModel EMA of weights; ema.module is evaluated and saved.
#
# Experiment 12 -- RESULT: REVERT (run_id 12, candidate 1e58069)
#   HELDOUTSCORE 1.632397 <= 1.632837 -> reverted to champion run 10 code.
#   S_KQ1..8 = 1.914 1.656 1.543 1.480 1.442 1.482 1.620 1.921.
#   TRAIN_S 529.6, INFERENCE_S 0.407.
#   Falsifier TRIGGERED (no gain; -0.00015 vs run 10). Interpretation: the
#     cosine-to-zero final iterate is already noise-free; the model is at its
#     capacity/data limit for this compute budget. Remaining levers: compute
#     efficiency (to afford more capacity/data) and beam structure.
#
# -----------------------------------------------------------------------------
# Experiment 13 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 10)
#   Hypothesis: within the fixed 4000-step budget the champion is SAMPLE-limited
#     (run 12 ruled out final-iterate noise; extra capacity in runs 7-8 gave only
#     +0.002 each). Spending freed compute on 25% more fresh channels per update
#     (batch 256 -> 320) improves generalization of the policy.
#   Necessary enabling parts (exact, function-preserving; verified pre-launch:
#     max |cos| deviation 1e-15 of dominant vectors, rate rel. diff 1e-7 from
#     float32 summation order; full-loop step 91 -> 70 ms):
#     * closed-form dominant left singular vector of the 2x8 direct channels via
#       the 2x2 Hermitian A A^H (replaces both batched SVDs);
#     * linear msg_out applied after the masked mean instead of per pair.
#   Falsifier: HELDOUTSCORE <= 1.632837 (champion + delta).
#
# Experiment 13 -- RESULT: REVERT (run_id 13, candidate 15285be)
#   HELDOUTSCORE 1.632603 (+0.000055 vs champion, < delta) -> reverted to run 10 code.
#   S_KQ1..8 = 1.914 1.657 1.544 1.481 1.442 1.482 1.621 1.921.
#   TRAIN_S 462.9 (speedups confirmed: batch 320 cheaper than batch 256 before),
#     INFERENCE_S 0.290.
#   Falsifier TRIGGERED. Interpretation: not sample-limited either. Final-iterate
#     noise (run 12), data (run 13) and moderate capacity (runs 7-8) are all
#     ruled out; the ceiling is the beam STRUCTURE. The only recent large wins
#     were receive-aware alternations (+0.0063, +0.0046; ratio ~0.73 -> still
#     converging). The exact speedups are reverted with the code by rule but
#     may be reused as enabling parts of later hypotheses.
#
# -----------------------------------------------------------------------------
# Experiment 14 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 10)
#   Hypothesis: the transmit/receive alternation is still unconverged after 2
#     steps (gains +0.0063, +0.0046; ratio ~0.73). Three further fixed
#     alternations (5 total; predeclared count, no grad, same learned weights)
#     bring the beams closer to the transmit/receive fixed point, and the
#     learned weights adapt to the converged geometry.
#   Necessary enabling part: the exact exp-13 speedups (closed-form 2x2 eigen
#     dominant vectors; msg_out after aggregation) to stay within 600 s.
#     Batch stays 256.
#   Falsifier: HELDOUTSCORE <= 1.632837 (champion + delta), OR gain < +0.002
#     (score <= 1.634548; a 0.73-ratio continuation predicts ~+0.006).
#   Change: N_EXTRA_ALT = 3 alternation loop; speedups.
#
# Experiment 14 -- RESULT: COMMIT (run_id 14, candidate 232a085) -> global champion
#   HELDOUTSCORE 1.635549 (champion 1.632548 + delta -> banked).
#   S_KQ1..8 = 1.916 1.659 1.546 1.483 1.445 1.486 1.625 1.923.
#   TRAIN_S 459.6 (speedups more than pay for 3 alternations), INFERENCE_S 0.460.
#   Falsifier NOT triggered (+0.0030 >= +0.002), but well below the +0.006
#     geometric extrapolation: alternations converge faster than ratio 0.73;
#     further fixed alternations are unlikely to add much. Gains uniform over Kq.
#
# -----------------------------------------------------------------------------
# Experiment 15 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 14)
#   Hypothesis: the stage-2 refinement chooses its corrections from the
#     PRE-alternation operating point (SE under un-alternated stage-1 beams and
#     edge features of the dominant-singular-vector geometry), while the final
#     beams live on the alternated receive-aware geometry. Reordering -- first 3
#     fixed alternations of the stage-1 beamformer, then refinement that observes
#     SE/rank under those alternated beams AND edge features recomputed from the
#     alternated geometry, then 2 fixed alternations of the refined beamformer
#     (5 total, unchanged) -- matches the corrections to the operating point
#     they act on.
#   Falsifier: HELDOUTSCORE <= 1.635838 (champion + delta).
#   Change: forward reorganized (alternate(), _edge_features()); N_ALT1=3,
#     N_ALT2=2 (N_EXTRA_ALT no longer used). Parameters unchanged (515k).
#
# Experiment 15 -- RESULT: COMMIT (run_id 15, candidate 00dcd65) -> global champion
#   HELDOUTSCORE 1.637546 (champion 1.635549 + delta -> banked).
#   S_KQ1..8 = 1.920 1.661 1.549 1.486 1.448 1.488 1.625 1.923.
#   TRAIN_S 486.7, INFERENCE_S 0.507.
#   Falsifier NOT triggered (+0.0020). Gain concentrated at small/mid Kq
#     (Kq=1 +0.0035, Kq=8 ~0): refinement decisions now match the operating
#     point; allocation for the weakest users benefits most.
#
# -----------------------------------------------------------------------------
# Experiment 16 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 15)
#   Hypothesis: the per-BS softmax forces every BS to radiate exactly Pmax. When
#     the neighbouring cell's weak (cell-edge) users are binding -- typical for
#     Kq=1-2 -- excess power on strong users is pure inter-cell interference that
#     RZF weights can only partly null. A learned per-BS power backoff (5th
#     "unused" softmax slot; logit = -3 + cell-mean of a zero-init per-node head
#     on the final hidden states; used by the final alternations and beams)
#     improves the small-Kq tasks.
#   Falsifier: HELDOUTSCORE <= 1.637835 (champion + delta), OR
#     mean(S_KQ1, S_KQ2) <= 1.7903 (run 15 value).
#   Change: structured_beams(null_logit); null_head (129 params); NULL_LOGIT0.
#
# Experiment 16 -- RESULT: REVERT (run_id 16, candidate 6c7c01a)
#   HELDOUTSCORE 1.636955 <= 1.637835 -> reverted to champion run 15 code.
#   S_KQ1..8 = 1.919 1.661 1.548 1.485 1.448 1.487 1.625 1.923.
#   TRAIN_S 469.2, INFERENCE_S 0.504.
#   Falsifier TRIGGERED (mean S_KQ1..2 = 1.7898 < 1.7903). Interpretation: full
#     per-BS power is effectively optimal here -- cell-edge users are largely
#     noise-limited and the learned RZF weights already handle inter-cell
#     interference; the backoff slot only costs the init waste.
#
# -----------------------------------------------------------------------------
# Experiment 17 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 15)
#   Hypothesis: the GNN champion is optimization-PATH-limited: the update count
#     is capped at 4000, and neither more data per update (run 13) nor iterate
#     averaging (run 12) helped, so the distance Adam travels from the SLNR
#     initialization -- set by the peak learning rate under the cosine
#     schedule -- limits the reachable solution. Doubling the peak LR to 2e-3
#     (cosine to 0 unchanged; the LR was inherited untuned from the MLP seed)
#     reaches a better solution.
#   Falsifier: HELDOUTSCORE <= 1.637835 (champion + delta), OR any non-finite
#     training loss.
#   Change: LR 1e-3 -> 2e-3.
#
# Experiment 17 -- RESULT: COMMIT (run_id 17, candidate 0480fac) -> global champion
#   HELDOUTSCORE 1.640124 (champion 1.637546 + delta -> banked).
#   S_KQ1..8 = 1.921 1.663 1.553 1.490 1.453 1.491 1.626 1.923.
#   TRAIN_S 483.3, INFERENCE_S 0.489. Training stable.
#   Falsifier NOT triggered (+0.0026). Interpretation: optimization-path-limited
#     confirmed; mid-Kq gains largest (Kq 4-5 +0.004..0.005), Kq=8 flat.
#     The LR optimum may lie higher still.
#
# -----------------------------------------------------------------------------
# Experiment 18 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 17)
#   Hypothesis: the learning-rate optimum lies above 2e-3: run 17 showed the
#     trajectory is still too short at 2e-3, so another doubling of the peak LR
#     to 4e-3 (cosine to 0 unchanged) again reaches a better solution without
#     destabilizing training (all outputs pass through the bounded structured
#     layer with clamped log q).
#   Falsifier: HELDOUTSCORE <= 1.640413 (champion + delta), OR any non-finite
#     training loss.
#   Change: LR 2e-3 -> 4e-3.
#
# Experiment 18 -- RESULT: FAIL (run_id 18, candidate e8590f8)
#   Training-time violation: TRAIN_S 600.103 > 600 s after ~3630/4000 steps
#     (~165 ms/step vs 121 ms/step for the identical-cost run 17) -> FAIL / no
#     score, reverted to champion run 17 code.
#   Falsifier: not evaluable (no score). Diagnostic: the logged training loss
#     at LR 4e-3 was already worse than run 17 at matching steps (~-1.62 vs
#     ~-1.70 near step 3200-3300), so 4e-3 likely overshoots the LR optimum.
#   Slowdown cause unresolved (same per-step FLOPs; candidates: concurrent
#     machine load or slow numerics from larger weights). Follow-up rule: keep
#     projected TRAIN_S <= ~500 s and check machine speed before each launch.
#
# -----------------------------------------------------------------------------
# Experiment 19 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 17)
#   Hypothesis: the champion is limited by optimization PATH length (run 17),
#     but the peak LR cannot rise much above 2e-3 (run 18: worse loss already
#     early at 4e-3). Plain cosine spends much of the budget at small LR
#     (integrated LR 0.5*peak*N). A trapezoid schedule at the same 2e-3 peak --
#     constant for the first 50% of steps, then cosine to 0 -- gives 1.5x the
#     integrated LR, travelling further without a higher peak, still annealed.
#   Falsifier: HELDOUTSCORE <= 1.640413 (champion + delta).
#   Change: LambdaLR trapezoid (LR_HOLD_FRAC = 0.5) replaces CosineAnnealingLR.
#   Pre-launch machine-speed check: 92.5 ms/step (normal).
#
# Experiment 19 -- RESULT: REVERT (run_id 19, candidate 2c96374)
#   HELDOUTSCORE 1.640257 (+0.00013 vs champion, < delta) -> reverted to run 17 code.
#   S_KQ1..8 = 1.922 1.664 1.552 1.491 1.452 1.491 1.626 1.923.
#   TRAIN_S 476.0, INFERENCE_S 0.500.
#   Falsifier TRIGGERED. Interpretation: at a 2e-3 peak, extra integrated LR no
#     longer helps; together with run 18 the optimizer is roughly tuned. The
#     remaining ceiling is again the structure / information of the policy.
#
# -----------------------------------------------------------------------------
# Experiment 20 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 17)
#   Hypothesis: the stage-2 refinement chooses its corrections from channel-
#     GEOMETRY proxies on its edges (effective-channel correlations, large-scale
#     cross gains), whereas what those corrections trade off is the interference
#     each stream actually causes at each receiver AFTER beamforming and MMSE
#     combining at the current operating point. Adding the actual post-combining
#     interference-to-noise ratios between every ordered user pair under the
#     alternated stage-1 beamformer (both directions, log10(INR+1e-6)/4, no
#     grad) to the stage-2 edge features lets the refinement target the binding
#     interference links.
#   Falsifier: HELDOUTSCORE <= 1.640413 (champion + delta).
#   Change: EDGE_FEAT2 = 7 for msg2_e; INR computed from the stage-1 combiners.
#   Pre-launch machine-speed check: 93.9 ms/step.
#
# Experiment 20 -- RESULT: COMMIT (run_id 20, candidate c44c7b5) -> global champion
#   HELDOUTSCORE 1.642017 (champion 1.640124 + delta -> banked).
#   S_KQ1..8 = 1.923 1.665 1.554 1.494 1.456 1.494 1.628 1.923.
#   TRAIN_S 472.9, INFERENCE_S 0.503.
#   Falsifier NOT triggered (+0.0019). Gains largest at mid Kq (4-6: ~+0.003):
#     the actual post-combining interference graph is more useful to the
#     refinement than geometry proxies.
#
# -----------------------------------------------------------------------------
# Experiment 21 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 20)
#   Hypothesis: run 20 showed the actual post-combining interference graph beats
#     geometry proxies for the refinement stage, but the stage-1 GNN -- which
#     makes the bulk of the power/regularization decisions -- still sees only
#     geometry. Adding the same post-combining INR edge features to stage 1
#     (both directions, under the SLNR-init beamformer and its MMSE combiners,
#     no grad) improves stage-1 decisions, again mostly at mid Kq.
#   Falsifier: HELDOUTSCORE <= 1.642306 (champion + delta).
#   Change: msg_e input EDGE_FEAT2; post_combining_inr() helper shared by both
#     stages (stage-2 features numerically unchanged).
#   Pre-launch machine-speed check: 94.3 ms/step.
#
# Experiment 21 -- RESULT: REVERT (run_id 21, candidate f3e5185)
#   HELDOUTSCORE 1.641146 <= 1.642306 -> reverted to champion run 20 code.
#   S_KQ1..8 = 1.922 1.664 1.553 1.492 1.454 1.492 1.628 1.924.
#   TRAIN_S 481.9, INFERENCE_S 0.520.
#   Falsifier TRIGGERED. Interpretation: at the generic SLNR-init operating
#     point the interference graph carries no information beyond geometry; it
#     only pays off at the LEARNED operating point (stage 2). Effort should go
#     into the refinement side.
#
# -----------------------------------------------------------------------------
# Experiment 22 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 20)
#   Hypothesis: since runs 15/20 the refinement stage receives rich operating-
#     point information (SE/rank, alternated geometry, post-combining
#     interference graph) but processes it with a single message-passing hop, so
#     a correction for user i can only account for its direct interferers, not
#     for how changing THEIR power/regularization affects their own binding
#     links. A second refinement hop (GNN_LAYERS2 1 -> 2) coordinates the
#     corrections across the interference graph, mostly helping mid Kq.
#   Falsifier: HELDOUTSCORE <= 1.642306 (champion + delta).
#   Change: GNN_LAYERS2 = 2.
#
# Experiment 22 -- RESULT: COMMIT (run_id 22, candidate 00805c6) -> global champion
#   HELDOUTSCORE 1.642979 (champion 1.642017 + delta -> banked).
#   S_KQ1..8 = 1.923 1.666 1.556 1.495 1.457 1.494 1.628 1.924.
#   TRAIN_S 536.3 (only ~64 s headroom: later changes must be compute-neutral),
#     INFERENCE_S 0.552.
#   Falsifier NOT triggered (+0.00096; small, spread over Kq 1-6).
#
# -----------------------------------------------------------------------------
# Experiment 23 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 22)
#   Hypothesis: each refinement node sees its operating-point SE, which merges
#     desired-signal strength and total interference into one number, but the
#     right correction depends on WHICH binds: a noise-limited user needs more
#     power, an interference-limited user needs its interferers to null harder.
#     Adding the user's post-combining desired SNR and its total post-combining
#     INR (log10(.)/4; from the already computed powers, compute-neutral) as
#     stage-2 node features lets the refinement choose the right correction.
#   Falsifier: HELDOUTSCORE <= 1.643268 (champion + delta).
#   Change: NODE_FEAT2 4 -> 6 (enc2 input).
#   PRE-LAUNCH ADDENDUM (exp 23, before launch; hypothesis/falsifier unchanged):
#     machine timing became noisy (107-115 ms/step -> risk of > 600 s), so exact
#     function-preserving speedups were added as a necessary enabling part:
#     vectorized training physics (diagonal extraction instead of per-(k,b)
#     loops) and |x|^2 = Re^2 + Im^2 / cnorm instead of complex abs/vector_norm.
#     Verified pre-launch: training rates vs evaluator max diff 6e-14 Mbps,
#     model rates rel. diff 5e-13, gradients identical; 107 -> 87 ms/step.
#
# Experiment 23 -- RESULT: REVERT (run_id 23, candidate f1c0400)
#   HELDOUTSCORE 1.641661 <= 1.643268 -> reverted to champion run 22 code.
#   S_KQ1..8 = 1.923 1.665 1.554 1.493 1.455 1.493 1.627 1.923.
#   TRAIN_S 445.6 (exact physics vectorization confirmed: -17% train time),
#     INFERENCE_S 0.395.
#   Falsifier TRIGGERED. Interpretation: the refinement already infers noise-
#     vs interference-limited status from SE + the INR edges; explicit node
#     summaries add nothing. Note: GNN run-to-run perturbations from small
#     changes (init/data order) appear to be ~1e-3, larger than the MLP-
#     calibrated delta, so +/-0.001 effects are near the noise floor.
#
# -----------------------------------------------------------------------------
# Experiment 24 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 22)
#   Hypothesis: the productive line has been refinement = message passing over
#     features observed at the CURRENT operating point (runs 8, 15, 20, 22), but
#     it is truncated after one round: the stage-2 corrections move the operating
#     point again and nothing observes/corrects that new point. A third,
#     weight-separate refinement stage (deep unfolding: observe SE/rank,
#     alternated geometry and post-combining INR graph under the stage-2
#     beamformer -> 1 message-passing hop -> zero-init additive correction ->
#     1 more fixed alternation; 6 alternations total) captures further gain.
#   Necessary enabling parts (exact up to float rounding; verified pre-launch):
#     exp-23 physics vectorization (rates vs evaluator 6e-14 Mbps) and masked
#     means as broadcast multiply-sum (model rate rel. diff 1.2e-7);
#     stage-2 features now come from a shared observe() helper (same math).
#     Step 106.8 -> 99.9 ms; projected TRAIN_S ~510 s.
#   Falsifier: HELDOUTSCORE <= 1.643268 (champion + delta).
#   Change: enc3/msg3*/upd3/head3 (GNN_LAYERS3=1), N_ALT3=1; 782k params.
#
# Experiment 24 -- RESULT: REVERT (run_id 24, candidate beddf2d)
#   HELDOUTSCORE 1.643194 (+0.00022 vs champion, < delta) -> reverted to run 22 code.
#   S_KQ1..8 = 1.924 1.667 1.557 1.496 1.457 1.493 1.628 1.924.
#   TRAIN_S 522.2, INFERENCE_S 0.500.
#   Falsifier TRIGGERED. Interpretation: refinement depth has saturated; a third
#     observe/correct round adds nothing measurable. Recent experiments (20-24)
#     move the score by ~1e-3 either way, i.e. at the GNN run-to-run noise level.
#
# -----------------------------------------------------------------------------
# Experiment 25 -- PRE-REGISTRATION (family interference_gnn, depth on champion run 22)
#   Hypothesis: all five GNN stages (3 stage-1 + 2 refinement hops) share a
#     128-wide hidden state that must encode each user's interference
#     neighbourhood, Kq-relative status and operating-point summaries; recent
#     information/refinement additions produced small saturating gains,
#     consistent with a representational bottleneck rather than missing
#     information. Widening the hidden state 128 -> 160 (~1.56x params) raises
#     the score.
#   Necessary enabling parts (exact up to float rounding, as verified in exps
#     23/24): physics vectorization + masked means as broadcast multiply-sum.
#     Step 80 ms (width 128) -> 96.7 ms (width 160); projected TRAIN_S ~495 s.
#   Falsifier: HELDOUTSCORE <= 1.643268 (champion + delta).
#   Change: GNN_HIDDEN 128 -> 160.
#
# Experiment 25 -- RESULT: COMMIT (run_id 25, candidate 7439053) -> global champion
#   HELDOUTSCORE 1.644191 (champion 1.642979 + delta -> banked).
#   S_KQ1..8 = 1.925 1.667 1.557 1.496 1.459 1.496 1.630 1.924.
#   TRAIN_S 487.0, INFERENCE_S 0.454.
#   Falsifier NOT triggered (+0.0012). Width was a modest bottleneck; width 192
#     would project to ~600 s and is not affordable.
#
# =============================================================================
# CAMPAIGN EARLY STOP (charter §13 early-stop rule) -- after experiment 25
#   Stopping experiment number: 25 (25 of 50 launched; run_ids 1-25 all logged).
#   Global champion (campaign winner): run 25, family interference_gnn,
#     candidate 7439053, banked a3fa5e8, HELDOUTSCORE 1.644191
#     (S_KQ1..8 = 1.925 1.667 1.557 1.496 1.459 1.496 1.630 1.924).
#   Main directions tested:
#     * centralized_mlp: structured RZF/SLNR beam head (r1), score-aligned
#       per-Kq normalized loss (r2), explicit link-quality features (r3),
#       4000 cosine steps (r4), batch 256 (r5) -> 1.503005.
#     * interference_gnn (family 2, promoted after its 5-run window r6-r10):
#       equivariant message passing (r6), rank features (r7), observe/correct
#       refinement stage (r8), receive-aware effective channels and fixed
#       transmit/receive alternations (r9, r10, r14), refinement at the
#       alternated operating point (r15), LR 2e-3 (r17), post-combining
#       interference-graph edges (r20), two-hop refinement (r22), width 160 (r25).
#     * Saturated / rejected: soft membership loss (r11), weight EMA (r12),
#       more data per step (r13), per-BS power backoff (r16), LR 4e-3 (r18 FAIL,
#       time limit), trapezoid LR (r19), stage-1 INR edges (r21), SNR/INR node
#       features (r23), third refinement stage (r24).
#   Reason for stopping: further experiments are unlikely to produce meaningful
#     progress. (i) Post-hoc diagnostics of the banked model on TRAINING-stream
#     channels (seed 789; selection/final untouched) show near-perfect threshold
#     equalization: r(Kq)/r(Kq+1) median 0.992-0.998 with 4-5 users within 5% of
#     the threshold, i.e. allocation is near-optimal within the structure;
#     (ii) beam directions have converged: extra fixed alternations without
#     retraining change the mean ratio by <= 0.001 and hurt small Kq; the
#     RZF/optimal-structure family is theoretically complete for the effective
#     channels; (iii) data, averaging, LR schedule, refinement depth and further
#     features are saturated, and the next affordable width step (192) projects
#     to ~600 s training; (iv) null-effect changes move the GNN score by about
#     +/-0.001 (runs 12, 13, 19, 21, 23, 24), ~3x the MLP-calibrated delta, so
#     further marginal "wins" would largely fit selection-set noise and would
#     not be expected to transfer to the protected final set.
#   last_model.pt currently holds the run-25 champion weights (runtime artifact;
#     Git + this changelog + log.csv are authoritative). No post-campaign tuning.
