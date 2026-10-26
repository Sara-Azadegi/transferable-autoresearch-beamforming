"""
train.py -- ONLY mutable research source file for Direct 7-cell Autoresearch.

Deliberately generic Path-A seed
--------------------------------
This seed contains no Baby champion, Baby experiment log, or Baby-discovered
architecture. It is a plain centralized full-CSI residual MLP anchored at the
frozen evaluator's equal-power direct-channel-SVD baseline.

Seed recipe:
  * input = asinh(Re H/sigma), asinh(Im H/sigma), unit-norm SVD-baseline beams
    (Re, Im), and Kq/35;
  * MLP 8401 -> 128 -> 128 -> 560, ReLU hidden activations;
  * final layer zero-initialized, so the untrained seed equals the fixed baseline;
  * output = baseline + learned complex correction, followed only by per-BS
    rescaling when necessary to satisfy the shared 43-dBm BS budget;
  * exact hard network-wide SLqP training loss;
  * Adam, lr=1e-3, 2000 steps, batch=16, fresh channels each update;
  * Kq ~ Uniform{1,...,35}; fixed TORCH_SEED=2027 and SEED_TRAIN=789;
  * CPU; float32 NN, complex128/float64 differentiable training physics;
  * only the final checkpoint is judged by the immutable process-separated director pipeline.

During Autoresearch, the agent may redesign this file subject to program.md.
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import torch
import torch.nn as nn

torch.set_num_threads(20)

from prepare7 import (
    B,
    BANDWIDTH_HZ,
    K_TOTAL,
    K_USERS,
    M_TX,
    N_RX,
    NOISE_POWER_W,
    P_MAX_W,
    SEED_TRAIN,
    TORCH_SEED,
    TrainingChannelStream,
)

# =============================================================================
# Frozen Path-A starting recipe
# =============================================================================
DEVICE = torch.device("cpu")
TRAIN_REAL_DTYPE = torch.float32
TRAIN_COMPLEX_DTYPE = torch.complex64
PHYS_COMPLEX_DTYPE = torch.complex128
PHYS_REAL_DTYPE = torch.float64

STEPS = 4000
BATCH = 40
LR = 2e-3       # run 18: peak LR 1e-3 -> 2e-3
LR_MIN = 1e-5  # run 4: cosine annealing floor
WARMUP_STEPS = 200  # run 18: linear warmup
FAMILY = "user_transformer_dualbf"
TRAIN_TIME_LIMIT_S = 600.0
RUN_TORCH_SEED = TORCH_SEED
RUN_TRAIN_SEED = SEED_TRAIN

CHANNEL_REAL_DIM = 2 * K_USERS * B * B * N_RX * M_TX   # 7840
BASELINE_REAL_DIM = 2 * K_USERS * B * M_TX             # 560
INPUT_DIM = CHANNEL_REAL_DIM + BASELINE_REAL_DIM + 1    # 8401
OUTPUT_DIM = 2 * K_USERS * B * M_TX                     # 560
HIDDEN = 128

STREAM_AMP = math.sqrt(P_MAX_W / K_USERS)


def baseline_beams(Hc: torch.Tensor) -> torch.Tensor:
    """Frozen equal-power dominant-direct-channel-SVD baseline."""
    direct = torch.stack(
        [Hc[:, :, b, b] for b in range(B)], dim=2
    )  # [batch,K,B,N,M]
    _, _, Vh = torch.linalg.svd(direct, full_matrices=False)
    return STREAM_AMP * Vh[..., 0, :].conj()


class Direct7SeedMLP(nn.Module):
    """Plain centralized full-CSI seed for Path A."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, OUTPUT_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _kq_vector(
        Kq: int | torch.Tensor,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
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

    def forward(
        self,
        H: torch.Tensor,
        Kq: int | torch.Tensor,
    ) -> torch.Tensor:
        expected_tail = (K_USERS, B, B, N_RX, M_TX)
        if H.ndim != 6 or tuple(H.shape[1:]) != expected_tail:
            raise ValueError(
                f"H shape {tuple(H.shape)} is incompatible with Direct-7C"
            )
        if not torch.is_complex(H):
            raise ValueError("H must be complex")

        Hc = H.to(device=DEVICE, dtype=TRAIN_COMPLEX_DTYPE)
        batch = Hc.shape[0]
        kq = self._kq_vector(Kq, batch, Hc.device)

        sigma = math.sqrt(NOISE_POWER_W)
        Hs = Hc / sigma
        xr = torch.asinh(Hs.real).reshape(batch, -1)
        xi = torch.asinh(Hs.imag).reshape(batch, -1)
        xk = (kq.to(TRAIN_REAL_DTYPE) / float(K_TOTAL)).unsqueeze(1)

        V0 = baseline_beams(Hc)
        v0n = V0 / STREAM_AMP
        x = torch.cat(
            (
                xr,
                xi,
                v0n.real.reshape(batch, -1),
                v0n.imag.reshape(batch, -1),
                xk,
            ),
            dim=1,
        )

        raw = self.net(x).reshape(batch, K_USERS, B, M_TX, 2)
        z = torch.complex(raw[..., 0], raw[..., 1])
        V = V0 + STREAM_AMP * z

        # Feasibility map only: no repair occurs in the evaluator.
        bs_pow = V.abs().square().sum(dim=(1, 3), keepdim=True)  # [T,1,B,1]
        scale = torch.clamp(
            torch.sqrt(P_MAX_W / (bs_pow + 1e-30)),
            max=1.0,
        )
        return V * scale


# =============================================================================
# Differentiable TRAINING copy of frozen 7-cell physics
# =============================================================================
def training_rates_mbps(H: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return training_rates_and_combiners(H, V)[0]


def training_rates_and_combiners(
    H: torch.Tensor, V: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    H = H.to(device=DEVICE, dtype=PHYS_COMPLEX_DTYPE)
    V = V.to(device=DEVICE, dtype=PHYS_COMPLEX_DTYPE)
    batch = H.shape[0]

    S = torch.einsum("tkbcnm,tjcm->tkbjcn", H, V)

    # desired[t,k,b,:] = S[t,k,b,k,b,:] (vectorized diagonal gather, run 7)
    desired = S.diagonal(dim1=1, dim2=3).diagonal(dim1=1, dim2=2).permute(0, 2, 3, 1)

    eye = torch.eye(
        N_RX,
        dtype=PHYS_COMPLEX_DTYPE,
        device=DEVICE,
    ).view(1, 1, 1, N_RX, N_RX)
    Ryy = torch.einsum("tkbjcn,tkbjcm->tkbnm", S, S.conj())
    Ryy = Ryy + NOISE_POWER_W * eye

    u = torch.linalg.solve(Ryy, desired.unsqueeze(-1)).squeeze(-1)

    proj = torch.einsum("tkbn,tkbjcn->tkbjc", u.conj(), S)
    stream_power = proj.abs().square()
    signal = stream_power.diagonal(dim1=1, dim2=3).diagonal(dim1=1, dim2=2)
    interference = stream_power.sum(dim=(-1, -2)) - signal

    noise_out = NOISE_POWER_W * u.abs().square().sum(dim=-1)
    denominator = interference + noise_out
    zero_signal = signal == 0
    safe_denominator = torch.where(
        zero_signal, torch.ones_like(denominator), denominator
    )
    sinr = signal / safe_denominator
    rates = torch.log2(1.0 + sinr) * (BANDWIDTH_HZ / 1e6)
    if not bool(torch.isfinite(rates).all()):
        raise RuntimeError("training physics produced non-finite user rate")
    return rates, u


def training_slqp(
    rates_mbps: torch.Tensor,
    Kq: torch.Tensor,
) -> torch.Tensor:
    flat = rates_mbps.reshape(rates_mbps.shape[0], K_TOTAL)
    sorted_rates = torch.sort(flat, dim=-1).values
    prefix = torch.cumsum(sorted_rates, dim=-1)
    return prefix.gather(1, (Kq - 1).unsqueeze(1)).squeeze(1)


# Run 14: annealed entropic smoothing of the bottom-Kq selection.
SMOOTH_TAU0 = 0.3          # natural-log-rate units
SMOOTH_END_FRAC = 0.75     # tau reaches 0 here; exact hard SLqP afterwards
SMOOTH_BISECT = 60
AUX_WEIGHTS = (0.5, 0.7, 0.85)  # deep supervision of intermediate stages (runs 15-17)


@torch.no_grad()
def smooth_cut_weights(
    rates_mbps: torch.Tensor, Kq: torch.Tensor, tau: float
) -> torch.Tensor:
    """w in [0,1]^35 with sum(w) = Kq: w_i = sigmoid((theta - log r_i)/tau).

    Entropy-regularized LP relaxation of the bottom-Kq selection; by Danskin,
    grad of sum(w*r) w.r.t. r is w. tau <= 0 gives the exact hard selection."""
    T = rates_mbps.shape[0]
    flat = rates_mbps.reshape(T, K_TOTAL)
    if tau <= 0.0:
        rank = torch.argsort(torch.argsort(flat, dim=1), dim=1)
        return (rank < Kq.unsqueeze(1)).to(flat.dtype)
    x = torch.log(flat + 1e-6)
    lo = x.min(dim=1).values - 40.0 * tau
    hi = x.max(dim=1).values + 40.0 * tau
    target = Kq.to(flat.dtype)
    for _ in range(SMOOTH_BISECT):
        mid = 0.5 * (lo + hi)
        tot = torch.sigmoid((mid.unsqueeze(1) - x) / tau).sum(dim=1)
        lo = torch.where(tot < target, mid, lo)
        hi = torch.where(tot < target, hi, mid)
    theta = 0.5 * (lo + hi)
    return torch.sigmoid((theta.unsqueeze(1) - x) / tau)


# Score-aligned normalization (introduced in run 1). The per-Kq baseline mean
# SLqP is estimated once from TRAINING-stream channels only.
NORM_CHANNELS = 1024
NORM_CHUNK = 256


@torch.no_grad()
def training_baseline_kq_means(channels: TrainingChannelStream) -> torch.Tensor:
    """[K_TOTAL] tensor: mean baseline SLqP for Kq=1..35 on training channels."""
    total = torch.zeros(K_TOTAL, dtype=PHYS_REAL_DTYPE)
    for _ in range(NORM_CHANNELS // NORM_CHUNK):
        H = channels.sample(NORM_CHUNK, dtype=torch.complex128).to(DEVICE)
        rates = training_rates_mbps(H, baseline_beams(H))
        flat = torch.sort(rates.reshape(NORM_CHUNK, K_TOTAL), dim=-1).values
        total += torch.cumsum(flat, dim=-1).sum(dim=0)
    return total / NORM_CHANNELS


# =============================================================================
# Family 2 (run 2): user-token Transformer + optimal dual-power beam structure
# =============================================================================
N_FEAT = 14
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128


@torch.no_grad()
def user_features(
    H: torch.Tensor,
    kq: torch.Tensor,
    Sv: torch.Tensor,
    Vh: torch.Tensor,
    gain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-user physically normalized token features [T,K,B,N_FEAT] (float32)."""
    T = H.shape[0]
    snr_scale = (P_MAX_W / K_USERS) / NOISE_POWER_W
    iso_scale = P_MAX_W / (M_TX * N_RX * NOISE_POWER_W)

    f_s1 = torch.log10(Sv[..., 0].square() * snr_scale + 1e-12) / 2.0
    f_s2 = torch.log10(Sv[..., 1].square() * snr_scale + 1e-12) / 2.0
    f_rel = f_s1 - f_s1.max(dim=1, keepdim=True).values

    if gain is None:
        gain = H.abs().square().sum(dim=(-1, -2))  # [T,K,B(serving),C(tx)]
    eyeB = torch.eye(B, dtype=PHYS_REAL_DTYPE).view(1, 1, B, B)
    own = (gain * eyeB).sum(dim=-1)
    inter = gain.sum(dim=-1) - own
    f_inr = torch.log10(inter * iso_scale + 1e-3) / 2.0
    leak = (gain * (1.0 - eyeB)).sum(dim=1).sum(dim=1)  # [T,C] power BS c leaks
    f_leak = (torch.log10(leak * iso_scale + 1e-3) / 2.0).unsqueeze(1).expand(T, K_USERS, B)

    # Baseline beams from the caller's direct-channel SVD (identical to
    # baseline_beams(H), without recomputing the SVD; run 17).
    r = training_rates_mbps(H, STREAM_AMP * Vh[..., 0, :].conj()) / (BANDWIDTH_HZ / 1e6)
    f_r = r
    f_logr = torch.log10(r + 1e-3)
    flat = r.reshape(T, K_TOTAL)
    rank = torch.argsort(torch.argsort(flat, dim=1), dim=1).reshape(T, K_USERS, B)
    rank = rank.to(PHYS_REAL_DTYPE)
    kqf = kq.to(PHYS_REAL_DTYPE).view(T, 1, 1).expand(T, K_USERS, B)
    f_rank = rank / (K_TOTAL - 1)
    f_kq = kqf / K_TOTAL
    f_in = (rank < kqf).to(PHYS_REAL_DTYPE)
    f_gap = (rank + 1.0 - kqf) / K_TOTAL
    crank = torch.argsort(torch.argsort(r, dim=1), dim=1).to(PHYS_REAL_DTYPE)
    f_crank = crank / (K_USERS - 1)

    v1 = Vh[..., 0, :]  # [T,K,B,M] unit-norm dominant directions
    gram = torch.einsum("tkbm,tjbm->tbkj", v1.conj(), v1).abs().square()
    gram = gram * (1.0 - torch.eye(K_USERS, dtype=PHYS_REAL_DTYPE))
    f_cmax = gram.max(dim=-1).values.transpose(1, 2)
    f_csum = gram.sum(dim=-1).transpose(1, 2)

    feats = torch.stack(
        (
            f_s1, f_s2, f_rel, f_inr, f_leak, f_r, f_logr,
            f_rank, f_kq, f_in, f_gap, f_crank, f_cmax, f_csum,
        ),
        dim=-1,
    )
    return feats.to(TRAIN_REAL_DTYPE)


class UserTransformerDualBF(nn.Module):
    """35 user tokens -> (dual uplink power q, downlink power share p) -> beams.

    v_{k,c} = sqrt(p_{k,c}) * unit( (I + sum_j q_j g_{j,c}^H g_{j,c} / sigma^2)^{-1}
                                    g_{k,c,c}^H ),
    with g_{j,c} = u_j^H H_{j,c} and u_j the dominant direct-channel receive vector.
    q -> 0 with equal shares reproduces the frozen SVD baseline exactly.
    Per-BS power: softmax over the 5 users plus one "unused power" slot.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(N_FEAT, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, FF_DIM, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(
            layer, N_LAYERS, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.q_log_scale = nn.Parameter(torch.zeros(()))
        self.null_bias = nn.Parameter(torch.tensor(-6.0))

    @staticmethod
    def _prep(H: torch.Tensor, Kq: int | torch.Tensor):
        expected_tail = (K_USERS, B, B, N_RX, M_TX)
        if H.ndim != 6 or tuple(H.shape[1:]) != expected_tail:
            raise ValueError(f"H shape {tuple(H.shape)} is incompatible with Direct-7C")
        if not torch.is_complex(H):
            raise ValueError("H must be complex")
        H = H.to(device=DEVICE, dtype=PHYS_COMPLEX_DTYPE)
        kq = Direct7SeedMLP._kq_vector(Kq, H.shape[0], H.device)
        return H, kq

    def _stage1(self, H: torch.Tensor, kq: torch.Tensor):
        T = H.shape[0]
        direct = torch.stack([H[:, :, b, b] for b in range(B)], dim=2)
        with torch.no_grad():
            U, Sv, Vh = torch.linalg.svd(direct, full_matrices=False)
        feats = user_features(H, kq, Sv, Vh)  # [T,K,B,F]

        tok = feats.transpose(1, 2).reshape(T, K_TOTAL, N_FEAT)  # cell-major
        z = self.norm(self.enc(self.inp(tok)))
        out = self.head(z).reshape(T, B, K_USERS, 3).transpose(1, 2)
        out = out.to(PHYS_REAL_DTYPE)  # [T,K,B,3]
        return feats, out, U[..., :, 0]

    def _beams(
        self, H: torch.Tensor, u: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        q = (P_MAX_W / K_USERS) * torch.exp(self.q_log_scale.double() + out[..., 0])
        g = torch.einsum("tkbn,tkbcnm->tkbcm", u.conj(), H) / math.sqrt(NOISE_POWER_W)
        eyeM = torch.eye(M_TX, dtype=PHYS_COMPLEX_DTYPE)
        A = eyeM + torch.einsum(
            "tkb,tkbcm,tkbcl->tcml", q.to(PHYS_COMPLEX_DTYPE), g.conj(), g
        )
        g_dir = torch.stack([g[:, :, c, c, :] for c in range(B)], dim=1)  # [T,C,K,M]
        w = torch.linalg.solve(A, g_dir.conj().transpose(-1, -2))  # [T,C,M,K]
        w = w.permute(0, 3, 1, 2)  # [T,K,C,M]
        w = w / torch.linalg.vector_norm(w, dim=-1, keepdim=True)

        null = self.null_bias.double() + out[..., 2].mean(dim=1, keepdim=True)
        logits = torch.cat((out[..., 1], null), dim=1)  # [T,K+1,B]
        share = torch.softmax(logits, dim=1)[:, :K_USERS]
        p = P_MAX_W * (1.0 - 1e-9) * share
        return torch.sqrt(p).unsqueeze(-1).to(PHYS_COMPLEX_DTYPE) * w

    def forward(self, H: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
        H, kq = self._prep(H, Kq)
        _, out, u = self._stage1(H, kq)
        return self._beams(H, u, out)


# Run 3: receive-aware second stage (fixed two stages, no utility-based selection).
N_FEAT2 = N_FEAT + 4 + 3


@torch.no_grad()
def stage_rate_features(r_mbps: torch.Tensor, kq: torch.Tensor) -> torch.Tensor:
    """[T,K,B,4]: log rate, network rank, in-cut indicator, rank gap to Kq."""
    T = r_mbps.shape[0]
    r = r_mbps / (BANDWIDTH_HZ / 1e6)
    rank = torch.argsort(torch.argsort(r.reshape(T, K_TOTAL), dim=1), dim=1)
    rank = rank.reshape(T, K_USERS, B).to(PHYS_REAL_DTYPE)
    kqf = kq.to(PHYS_REAL_DTYPE).view(T, 1, 1)
    return torch.stack(
        (
            torch.log10(r + 1e-3),
            rank / (K_TOTAL - 1),
            (rank < kqf).to(PHYS_REAL_DTYPE),
            (rank + 1.0 - kqf) / K_TOTAL,
        ),
        dim=-1,
    ).to(TRAIN_REAL_DTYPE)


class UserTransformerDualBF2Stage(UserTransformerDualBF):
    """Stage 1 as run 2; stage 2 rebuilds g with the MMSE combiners induced by the
    stage-1 beams and residually refines (q, p) from the achieved stage-1 rates."""

    def __init__(self) -> None:
        super().__init__()
        self.inp2 = nn.Linear(N_FEAT2, D_MODEL)
        layer2 = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, FF_DIM, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.enc2 = nn.TransformerEncoder(
            layer2, N_LAYERS, enable_nested_tensor=False
        )
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.head2 = nn.Linear(D_MODEL, 3)
        nn.init.zeros_(self.head2.weight)
        nn.init.zeros_(self.head2.bias)

    def forward(self, H: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
        H, kq = self._prep(H, Kq)
        T = H.shape[0]
        feats, out1, u1 = self._stage1(H, kq)
        V1 = self._beams(H, u1, out1)

        r1, u_mmse = training_rates_and_combiners(H, V1)
        u2 = u_mmse.detach()
        u2 = u2 / torch.linalg.vector_norm(u2, dim=-1, keepdim=True).clamp_min(1e-300)
        x2 = torch.cat(
            (feats, stage_rate_features(r1.detach(), kq), out1.to(TRAIN_REAL_DTYPE)),
            dim=-1,
        )
        tok = x2.transpose(1, 2).reshape(T, K_TOTAL, N_FEAT2)
        z = self.norm2(self.enc2(self.inp2(tok)))
        d = self.head2(z).reshape(T, B, K_USERS, 3).transpose(1, 2)
        out2 = out1 + d.to(PHYS_REAL_DTYPE)
        return self._beams(H, u2, out2)


# Run 5: relational interference-graph attention bias (zero-init).
N_PAIR = 4
REL_HIDDEN = 16


@torch.no_grad()
def pair_features(
    H: torch.Tensor, Vh: torch.Tensor, gain: torch.Tensor | None = None
) -> torch.Tensor:
    """[T,35,35,N_PAIR] cell-major token pairs (i,j): log leakage from BS(i) to
    user j, its transpose, same-cell flag, intra-cell beam collision."""
    T = H.shape[0]
    iso = P_MAX_W / (M_TX * N_RX * NOISE_POWER_W)
    if gain is None:
        gain = H.abs().square().sum(dim=(-1, -2))  # [T,K,B,C]
    g_tok = gain.permute(0, 2, 1, 3).reshape(T, K_TOTAL, B)  # [T,j,C]
    cell = torch.arange(K_TOTAL) // K_USERS
    leak = g_tok[:, :, cell].transpose(1, 2)  # [T,i,j] = gain of j from BS(i)
    e1 = torch.log10(leak * iso + 1e-3) / 2.0
    e2 = e1.transpose(1, 2)
    same = (cell[:, None] == cell[None, :]).to(PHYS_REAL_DTYPE).expand(T, -1, -1)
    v1 = Vh[..., 0, :].permute(0, 2, 1, 3).reshape(T, K_TOTAL, M_TX)
    coll = torch.einsum("tim,tjm->tij", v1.conj(), v1).abs().square() * same
    return torch.stack((e1, e2, same, coll), dim=-1).to(TRAIN_REAL_DTYPE)


def _make_rel() -> nn.Module:
    rel = nn.Sequential(
        nn.Linear(N_PAIR, REL_HIDDEN), nn.ReLU(), nn.Linear(REL_HIDDEN, N_HEADS)
    )
    nn.init.zeros_(rel[-1].weight)
    nn.init.zeros_(rel[-1].bias)
    return rel


class UserTransformerDualBF2StageRel(UserTransformerDualBF2Stage):
    """Run-4 two-stage model plus per-head additive attention biases computed from
    the pairwise interference graph (separate bias MLPs per stage)."""

    def __init__(self) -> None:
        super().__init__()
        self.rel1 = _make_rel()
        self.rel2 = _make_rel()

    @staticmethod
    def _mask(rel: nn.Module, pf: torch.Tensor) -> torch.Tensor:
        return rel(pf).permute(0, 3, 1, 2)  # [T,heads,35,35]

    @staticmethod
    def _encode(enc: nn.TransformerEncoder, x: torch.Tensor, bias: torch.Tensor):
        """Explicit pre-norm encoder pass with additive attention bias, using the
        layers' own weights (avoids the NaN-producing eval fast path for 3-D
        float masks)."""
        T, L, D = x.shape
        dh = D // N_HEADS
        for layer in enc.layers:
            attn = layer.self_attn
            h = layer.norm1(x)
            qkv = nn.functional.linear(h, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = (
                t.reshape(T, L, N_HEADS, dh).transpose(1, 2)
                for t in qkv.chunk(3, dim=-1)
            )
            o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            x = x + attn.out_proj(o.transpose(1, 2).reshape(T, L, D))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        return x

    def forward(self, H: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
        H, kq = self._prep(H, Kq)
        T = H.shape[0]
        direct = torch.stack([H[:, :, b, b] for b in range(B)], dim=2)
        with torch.no_grad():
            U, Sv, Vh = torch.linalg.svd(direct, full_matrices=False)
        feats = user_features(H, kq, Sv, Vh)
        pf = pair_features(H, Vh)

        tok = feats.transpose(1, 2).reshape(T, K_TOTAL, N_FEAT)
        z = self.norm(self._encode(self.enc, self.inp(tok), self._mask(self.rel1, pf)))
        out1 = self.head(z).reshape(T, B, K_USERS, 3).transpose(1, 2)
        out1 = out1.to(PHYS_REAL_DTYPE)
        V = self._beams(H, U[..., :, 0], out1)

        out = out1
        for inp, enc, norm, head, rel in self._refiners():
            r, u_mmse = training_rates_and_combiners(H, V)
            u = u_mmse.detach()
            u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1e-300)
            x = torch.cat(
                (feats, stage_rate_features(r.detach(), kq), out.to(TRAIN_REAL_DTYPE)),
                dim=-1,
            )
            tok_s = x.transpose(1, 2).reshape(T, K_TOTAL, N_FEAT2)
            z_s = norm(self._encode(enc, inp(tok_s), self._mask(rel, pf)))
            d = head(z_s).reshape(T, B, K_USERS, 3).transpose(1, 2)
            out = out + d.to(PHYS_REAL_DTYPE)
            V = self._beams(H, u, out)
        return V

    def _refiners(self):
        return [(self.inp2, self.enc2, self.norm2, self.head2, self.rel2)]


class UserTransformerDualBF3StageRel(UserTransformerDualBF2StageRel):
    """Run 6: untied third relational receive-aware refinement stage."""

    def __init__(self) -> None:
        super().__init__()
        self.inp3 = nn.Linear(N_FEAT2, D_MODEL)
        layer3 = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, FF_DIM, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.enc3 = nn.TransformerEncoder(
            layer3, N_LAYERS, enable_nested_tensor=False
        )
        self.norm3 = nn.LayerNorm(D_MODEL)
        self.head3 = nn.Linear(D_MODEL, 3)
        nn.init.zeros_(self.head3.weight)
        nn.init.zeros_(self.head3.bias)
        self.rel3 = _make_rel()

    def _refiners(self):
        return [
            (self.inp2, self.enc2, self.norm2, self.head2, self.rel2),
            (self.inp3, self.enc3, self.norm3, self.head3, self.rel3),
        ]


class UserTransformerDualBF3StageRelMu(UserTransformerDualBF3StageRel):
    """Run 13: run-7 model plus per-BS power-constraint multipliers mu_c,
    A_c = mu_c I + sum_j q_j g_{j,c}^H g_{j,c} / sigma^2 (KKT structure for
    per-BS power constraints). log mu_c is predicted residually per stage from
    the cell's pooled tokens (zero-init: starts exactly at the run-7 model)."""

    def __init__(self) -> None:
        super().__init__()
        self.mu1 = nn.Linear(D_MODEL, 1)
        self.mu2 = nn.Linear(D_MODEL, 1)
        self.mu3 = nn.Linear(D_MODEL, 1)
        for lin in (self.mu1, self.mu2, self.mu3):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _mu_heads(self):
        return (self.mu2, self.mu3)

    @staticmethod
    def _cell_pool(z: torch.Tensor) -> torch.Tensor:
        T = z.shape[0]
        return z.reshape(T, B, K_USERS, D_MODEL).mean(dim=2)  # [T,C,D]

    def _beams_mu(self, H, u, out, log_mu):
        q = (P_MAX_W / K_USERS) * torch.exp(self.q_log_scale.double() + out[..., 0])
        g = torch.einsum("tkbn,tkbcnm->tkbcm", u.conj(), H) / math.sqrt(NOISE_POWER_W)
        eyeM = torch.eye(M_TX, dtype=PHYS_COMPLEX_DTYPE)
        mu = torch.exp(log_mu).to(PHYS_COMPLEX_DTYPE)[..., None, None]  # [T,C,1,1]
        A = mu * eyeM + torch.einsum(
            "tkb,tkbcm,tkbcl->tcml", q.to(PHYS_COMPLEX_DTYPE), g.conj(), g
        )
        g_dir = torch.stack([g[:, :, c, c, :] for c in range(B)], dim=1)
        w = torch.linalg.solve(A, g_dir.conj().transpose(-1, -2))
        w = w.permute(0, 3, 1, 2)
        w = w / torch.linalg.vector_norm(w, dim=-1, keepdim=True)

        null = self.null_bias.double() + out[..., 2].mean(dim=1, keepdim=True)
        logits = torch.cat((out[..., 1], null), dim=1)
        share = torch.softmax(logits, dim=1)[:, :K_USERS]
        p = P_MAX_W * (1.0 - 1e-9) * share
        return torch.sqrt(p).unsqueeze(-1).to(PHYS_COMPLEX_DTYPE) * w

    def forward(self, H: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
        return self.forward_aux(H, Kq)[0]

    def forward_aux(self, H: torch.Tensor, Kq: int | torch.Tensor):
        """Returns (V_final, [rates of each intermediate stage's beams])."""
        H, kq = self._prep(H, Kq)
        T = H.shape[0]
        aux_rates = []
        direct = torch.stack([H[:, :, b, b] for b in range(B)], dim=2)
        with torch.no_grad():
            U, Sv, Vh = torch.linalg.svd(direct, full_matrices=False)
            gain = H.abs().square().sum(dim=(-1, -2))
        feats = user_features(H, kq, Sv, Vh, gain)
        pf = pair_features(H, Vh, gain)

        tok = feats.transpose(1, 2).reshape(T, K_TOTAL, N_FEAT)
        z = self.norm(self._encode(self.enc, self.inp(tok), self._mask(self.rel1, pf)))
        out = self.head(z).reshape(T, B, K_USERS, 3).transpose(1, 2).to(PHYS_REAL_DTYPE)
        log_mu = self.mu1(self._cell_pool(z)).squeeze(-1).to(PHYS_REAL_DTYPE)
        V = self._beams_mu(H, U[..., :, 0], out, log_mu)

        for (inp, enc, norm, head, rel), mu_head in zip(
            self._refiners(), self._mu_heads()
        ):
            r, u_mmse = training_rates_and_combiners(H, V)
            aux_rates.append(r)
            u = u_mmse.detach()
            u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1e-300)
            x = torch.cat(
                (feats, stage_rate_features(r.detach(), kq), out.to(TRAIN_REAL_DTYPE)),
                dim=-1,
            )
            tok_s = x.transpose(1, 2).reshape(T, K_TOTAL, N_FEAT2)
            z_s = norm(self._encode(enc, inp(tok_s), self._mask(rel, pf)))
            d = head(z_s).reshape(T, B, K_USERS, 3).transpose(1, 2)
            out = out + d.to(PHYS_REAL_DTYPE)
            log_mu = log_mu + mu_head(self._cell_pool(z_s)).squeeze(-1).to(PHYS_REAL_DTYPE)
            V = self._beams_mu(H, u, out, log_mu)
        return V, aux_rates


class UserTransformerDualBF4StageRelMu(UserTransformerDualBF3StageRelMu):
    """Run 17: run-16 model plus an untied fourth relational stage (+ mu head)."""

    def __init__(self) -> None:
        super().__init__()
        self.inp4 = nn.Linear(N_FEAT2, D_MODEL)
        layer4 = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, FF_DIM, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.enc4 = nn.TransformerEncoder(
            layer4, N_LAYERS, enable_nested_tensor=False
        )
        self.norm4 = nn.LayerNorm(D_MODEL)
        self.head4 = nn.Linear(D_MODEL, 3)
        nn.init.zeros_(self.head4.weight)
        nn.init.zeros_(self.head4.bias)
        self.rel4 = _make_rel()
        self.mu4 = nn.Linear(D_MODEL, 1)
        nn.init.zeros_(self.mu4.weight)
        nn.init.zeros_(self.mu4.bias)

    def _refiners(self):
        return [
            (self.inp2, self.enc2, self.norm2, self.head2, self.rel2),
            (self.inp3, self.enc3, self.norm3, self.head3, self.rel3),
            (self.inp4, self.enc4, self.norm4, self.head4, self.rel4),
        ]

    def _mu_heads(self):
        return (self.mu2, self.mu3, self.mu4)


MODEL_CLASSES = {
    "Direct7SeedMLP": Direct7SeedMLP,
    "UserTransformerDualBF": UserTransformerDualBF,
    "UserTransformerDualBF2Stage": UserTransformerDualBF2Stage,
    "UserTransformerDualBF2StageRel": UserTransformerDualBF2StageRel,
    "UserTransformerDualBF3StageRel": UserTransformerDualBF3StageRel,
    "UserTransformerDualBF3StageRelMu": UserTransformerDualBF3StageRelMu,
    "UserTransformerDualBF4StageRelMu": UserTransformerDualBF4StageRelMu,
}
TRAIN_MODEL_CLASS = "UserTransformerDualBF4StageRelMu"


# =============================================================================
# Director re-score interface
# =============================================================================
def load_candidate(checkpoint_path: str = "last_model.pt") -> nn.Module:
    """Reconstruct the saved candidate without training or held-out access.

    Autoresearch may change the architecture, but this public function must remain
    available so immutable director_rescore.py can load the candidate in its isolated
    inference process; a separate scoring process never imports train.py.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    cls = MODEL_CLASSES[checkpoint.get("model_class", "Direct7SeedMLP")]
    model = cls().to(device=DEVICE, dtype=TRAIN_REAL_DTYPE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


# =============================================================================
# One fixed-budget seed experiment
# =============================================================================
def train_once() -> Tuple[nn.Module, dict]:
    torch.manual_seed(RUN_TORCH_SEED)

    print(f"FAMILY {FAMILY}")
    print(f"SEEDS torch={RUN_TORCH_SEED} train={RUN_TRAIN_SEED}")
    print(f"THREADS {torch.get_num_threads()}")
    print(f"TORCH_VERSION {torch.__version__}")

    model = MODEL_CLASSES[TRAIN_MODEL_CLASS]().to(device=DEVICE, dtype=TRAIN_REAL_DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, fused=True)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0 / WARMUP_STEPS, total_iters=WARMUP_STEPS
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=STEPS - WARMUP_STEPS, eta_min=LR_MIN
            ),
        ],
        milestones=[WARMUP_STEPS],
    )
    channels = TrainingChannelStream(seed=RUN_TRAIN_SEED)
    kq_gen = torch.Generator(device="cpu").manual_seed(RUN_TORCH_SEED)

    model.train()
    train_t0 = time.perf_counter()

    kq_norm = training_baseline_kq_means(channels)
    print(
        "TRAIN_BASELINE_KQ_MEANS "
        + " ".join(f"{v:.4f}" for v in kq_norm.tolist())
    )

    for step in range(1, STEPS + 1):
        H = channels.sample(BATCH, dtype=torch.complex128).to(DEVICE)
        Kq = torch.randint(
            1,
            K_TOTAL + 1,
            (BATCH,),
            generator=kq_gen,
            dtype=torch.long,
        )

        V, aux_rates = model.forward_aux(H, Kq)
        rates = training_rates_mbps(H, V)
        slqp = training_slqp(rates, Kq)
        tau = SMOOTH_TAU0 * max(0.0, 1.0 - step / (SMOOTH_END_FRAC * STEPS))

        # One batched (row-independent, hence identical) bisection for the final
        # and all auxiliary rate sets (run 17 speedup).
        all_r = [rates] + list(aux_rates)
        w_all = smooth_cut_weights(
            torch.cat([r.detach().reshape(BATCH, K_TOTAL) for r in all_r], dim=0),
            Kq.repeat(len(all_r)),
            tau,
        ).split(BATCH, dim=0)

        def smooth_norm_slqp(r: torch.Tensor, w_cut: torch.Tensor) -> torch.Tensor:
            surrogate = (w_cut * r.reshape(BATCH, K_TOTAL)).sum(dim=1)
            return (surrogate / kq_norm[Kq - 1]).mean()

        loss = -smooth_norm_slqp(rates, w_all[0])
        for a_w, r_aux, w_aux in zip(AUX_WEIGHTS, aux_rates, w_all[1:]):
            loss = loss - a_w * smooth_norm_slqp(r_aux, w_aux)

        if not torch.isfinite(loss):
            raise RuntimeError(f"training failed: non-finite loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_elapsed = time.perf_counter() - train_t0
        if train_elapsed > TRAIN_TIME_LIMIT_S:
            print(f"TRAIN_S {train_elapsed:.6f}")
            raise RuntimeError(
                f"training-time violation: {train_elapsed:.3f}s > "
                f"{TRAIN_TIME_LIMIT_S:.1f}s"
            )

        if step == 1 or step % 100 == 0 or step == STEPS:
            print(
                f"STEP {step:04d}/{STEPS}  NORM_LOSS {loss.item():.6f}  "
                f"MEAN_SLQP_MBPS {slqp.mean().item():.6f}"
            )

    train_s = time.perf_counter() - train_t0
    print(f"TRAIN_S {train_s:.6f}")

    checkpoint = {
        "model_class": TRAIN_MODEL_CLASS,
        "state_dict": model.state_dict(),
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "torch_seed": RUN_TORCH_SEED,
        "train_seed": RUN_TRAIN_SEED,
        "train_s": train_s,
        "family": FAMILY,
    }
    torch.save(checkpoint, "last_model.pt")
    print("CHECKPOINT last_model.pt")
    return model, {"train_s": train_s, "checkpoint": "last_model.pt"}


if __name__ == "__main__":
    train_once()


# =============================================================================
# APPEND-ONLY RESEARCH CHANGELOG
# =============================================================================
# Seed / pre-campaign:
#   Generic centralized 8401->128->128->560 residual MLP, zero-initialized at
#   the frozen equal-power direct-SVD baseline; exact hard-SLqP loss; Adam 1e-3;
#   2000 steps; batch 16 fresh 7-cell channels; uniform Kq in 1..35; CPU;
#   float32 network with complex128/float64 differentiable MMSE physics.
#
# During Autoresearch, append every hypothesis/result here. Never rewrite or
# delete prior entries.

# Run 0 - pre-campaign seed champion
# Hypothesis: establish the generic direct-7-cell starting point.
# Falsifier: N/A - seed run before autonomous Experiment 1.
# Code change: none; original centralized_mlp seed.
# Authoritative HELDOUTSCORE: 0.998314753317
# Inference_s: 1.075382
# Train_s: 59.789947
# Result: COMMIT as initial global champion.
# Interpretation: slightly below the frozen 1.0 baseline.

# Run 1 - centralized_mlp - score-aligned normalized loss
# Hypothesis: the seed fails to beat the baseline because its raw-Mbps SLqP loss
#   is dominated by high-Kq rate scale (Kq=35 ~300x Kq=1); replacing it with the
#   score-aligned loss -F_Kq/m_Kq (m_Kq = baseline mean SLqP estimated on 1024
#   TRAINING-stream channels) lets the same MLP improve on the champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 0.998814753317 (not > champion+0.0005).
# Code change: loss = -(F_Kq / m_Kq).mean(); everything else identical to run 0.
# Launch commit: 84e3e8b89ef1460f0e4679d6c748b23dbf99f81f
# Authoritative HELDOUTSCORE: 0.998318643070 (inference_s 1.238430, train_s 47.28)
# Falsifier outcome: TRIGGERED (+0.000004 vs champion) -> hypothesis falsified.
# Result: REVERT. Code restored to run-0 champion.
# Interpretation: the loss did re-weight tasks (S_1 0.9974->0.9990, high-Kq
#   ratios slightly down) but the MLP still learns essentially nothing beyond the
#   baseline. The bottleneck is the architecture/output parameterization, not
#   only objective weighting. Normalized loss is neutral-to-useful and remains a
#   candidate ingredient for structured families.

# Run 2 - OPEN FAMILY 2: user_transformer_dualbf (grace attempt 1/5)
# Structural limitation of centralized_mlp: one 8401-dim raw-CSI vector through a
#   128-wide bottleneck to an unconstrained 560-dim additive correction plus a
#   global rescale; no per-user weight sharing, no permutation structure, no
#   knowledge of the optimal-beamformer manifold. Runs 0-1 learned ~nothing.
# New family: 35 user tokens of physically normalized features (direct SNRs,
#   INR, BS leakage, baseline rate/rank, Kq cut indicator, intra-cell beam
#   collisions) -> 2-layer Transformer (d=64) -> per-user dual uplink power q and
#   downlink power share p; beams v = sqrt(p) unit((I + sum_j q_j g_j^H g_j /
#   sigma^2)^-1 g_k^H) with g = u^H H (u = dominant direct receive vector);
#   exact per-BS feasibility by softmax (+ unused-power slot). q->0 reproduces
#   the frozen baseline exactly. Score-aligned loss from run 1 kept as an
#   integral part (a Kq-conditioned power policy needs per-Kq equal weighting).
# Hypothesis: the structured user-token family beats the global champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 0.998814753317.
# Launch commit: 80b6cccdd8e17418c8fafc2fc0e4e26cf52aeea9
# Authoritative HELDOUTSCORE: 1.577643936175 (inference_s 3.509573, train_s 72.29)
# Falsifier outcome: NOT triggered (+0.5793 over global champion).
# Result: COMMIT as family-local champion (checkpoint champion_run2.pt). Global
#   champion stays run 0 until the family grace window closes (sec. 15).
# Interpretation: the optimal-structure output manifold is the dominant lever.
#   S_Kq is U-shaped (2.11 at Kq=1, min 1.34 near Kq=21, 2.21 at Kq=35).
#   Training loss plateaued near -1.5 by step ~300, so the family looks
#   structure/feature-limited rather than optimization-length-limited.

# Run 3 - user_transformer_dualbf (grace attempt 2/5) - receive-aware 2nd stage
# Hypothesis: the family is limited by its beam manifold: g = u^H H uses the
#   fixed dominant receive vector, ignoring that each 2-antenna MMSE receiver can
#   suppress interference. A fixed second stage that rebuilds g with the MMSE
#   combiners induced by the stage-1 beams and residually refines (q,p) from the
#   achieved stage-1 rates/ranks (second 2-layer token encoder, zero-init head)
#   will improve the family-local champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.578143936175 (family champion + 0.0005).
# Code change: new class UserTransformerDualBF2Stage (run-2 class refactored into
#   _prep/_stage1/_beams, verified bit-identical on champion_run2.pt);
#   training_rates_and_combiners returns MMSE combiners. Training recipe unchanged.
# Launch commit: fa36a9b03bb1454e74427bd58ed7e7bc24337da0
# Authoritative HELDOUTSCORE: 1.595421295867 (inference_s 4.854750, train_s 103.57)
# Falsifier outcome: NOT triggered (+0.017777 over family champion).
# Result: family-local COMMIT (checkpoint champion_run3.pt); global champion
#   still run 0 during the grace window.
# Interpretation: receive-awareness mainly helps mid-Kq (S_21 1.341->1.379,
#   S_22 1.342->1.375) where the tradeoff between strong and weak users is
#   hardest; Kq=1 slightly lower (2.108->2.091).

# Run 4 - user_transformer_dualbf (grace attempt 3/5) - training budget/annealing
# Hypothesis: the 2-stage family is optimization-noise-limited (batch-16 random-
#   Kq normalized-SLqP gradients are very noisy; constant lr=1e-3 leaves SGD
#   noise in the final weights). A 4x sample budget (batch 32, 4000 steps) with
#   cosine LR annealing 1e-3 -> 1e-5 will improve the family champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.595921295867.
# Code change: STEPS 2000->4000, BATCH 16->32, CosineAnnealingLR(eta_min=1e-5).
# Launch commit: 67dbc0079109d1e6be83cd9197adc169c459dc4a
# Authoritative HELDOUTSCORE: 1.623292287454 (inference_s 4.800955, train_s 289.07)
# Falsifier outcome: NOT triggered (+0.027871 over family champion).
# Result: family-local COMMIT (checkpoint champion_run4.pt).
# Interpretation: near-uniform gain over all Kq (S_1 2.091->2.170, S_22
#   1.375->1.391, S_35 2.221->2.224) confirms optimization noise was limiting.
#   STEPS is now at the 4000 cap; further budget can only come from batch size
#   (train_s 289 s leaves ~2x headroom).

# Run 5 - user_transformer_dualbf (grace attempt 4/5) - relational attention bias
# Hypothesis: the token encoder cannot see who interferes with whom (features are
#   per-user aggregates), yet dual-power/power-share decisions hinge on the
#   pairwise interference graph. A learned per-head additive attention bias from
#   pairwise features (log leakage BS(i)->user j and transpose, same-cell flag,
#   intra-cell beam collision), zero-initialized, will improve the family champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.623792287454.
# Code change: class UserTransformerDualBF2StageRel (separate bias MLPs 4->16->4
#   per stage). Encoder layers run through an explicit pre-norm path with
#   scaled_dot_product_attention using the layers' own weights, because the
#   PyTorch eval fast path returns NaN for 3-D float masks (verified: explicit
#   path == library encoder to 5e-7; train/eval outputs identical).
# Launch commit: af10e1c83cb3e1ae09803a0c3c52d0b384c46339
# Authoritative HELDOUTSCORE: 1.653898982260 (inference_s 5.230513, train_s 313.44)
# Falsifier outcome: NOT triggered (+0.030607 over family champion).
# Result: family-local COMMIT (checkpoint champion_run5.pt).
# Interpretation: interference-graph awareness helps most at mid Kq (S_22
#   1.391->1.440, S_18 1.433->1.473) and least at Kq=35 (2.224->2.227).

# Run 6 - user_transformer_dualbf (grace attempt 5/5) - third relational stage
# Hypothesis: receive-aware refinement is not exhausted after one pass: stage-2
#   beams induce different MMSE combiners/rates than stage-1 beams, so an untied
#   third relational stage (g rebuilt from stage-2 MMSE combiners, residual
#   (q,p) refinement from stage-2 rates/ranks, zero-init head) will improve the
#   family champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.654398982260.
# Code change: class UserTransformerDualBF3StageRel; the run-5 forward was
#   rewritten as a loop over refinement stages (verified bit-identical on
#   champion_run5.pt).
# Launch commit: 830fc5e48f5c781e269bec256c926c353efe659e
# Authoritative HELDOUTSCORE: 1.657249680599 (inference_s 6.144518, train_s 392.71)
# Falsifier outcome: NOT triggered (+0.003351 over family champion).
# Result: family-local COMMIT (checkpoint champion_run6.pt).
# Interpretation: small, broad gain (largest at mid Kq: S_22 1.440->1.445);
#   diminishing returns from extra refinement stages at +80 s training cost.
#
# FAMILY 2 GRACE WINDOW CLOSED after 5 launched attempts (runs 2-6).
#   S_family = 1.657249680599 > S_global + 0.0005 = 0.998814753317.
#   -> user_transformer_dualbf run 6 REPLACES the global champion.
#   Global champion: run 6, 1.657249680599, checkpoint champion_run6.pt,
#   launch commit 830fc5e48f5c781e269bec256c926c353efe659e.

# Run 7 - user_transformer_dualbf (global champion family) - batch 32->48
# Hypothesis: the run-6 champion is still sample/gradient-noise-limited (as run
#   4 showed for the 2-stage model); raising batch 32->48 at 4000 steps (1.5x
#   samples) improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.657749680599 (global champion + 0.0005).
# Code change: BATCH 48. Enabler (pure engineering): training physics per-(k,b)
#   Python loops replaced by torch.diagonal gathers; verified bit-identical
#   (rates, gradients and champion_run6 outputs: max diff 0.0); ~30% faster.
# Launch commit: 6f8253d8e2b50fa6cffeae1fb4eaa2f614da3535
# Authoritative HELDOUTSCORE: 1.661089301409 (inference_s 5.707297, train_s 371.39)
# Falsifier outcome: NOT triggered (+0.003840 over global champion).
# Result: COMMIT -> new global champion (checkpoint champion_run7.pt).
# Interpretation: more samples still help but with diminishing returns
#   (run 4: +0.028 for 4x samples; run 7: +0.004 for 1.5x). Batch 64 would
#   project to ~505 s, too close to the 600 s cap for this architecture.

# Run 8 - OPEN FAMILY 3: dualbf_wmmse_unrolled (grace attempt 1/5)
# Structural limitation of user_transformer_dualbf: every stage *predicts*
#   (q,p) in one shot, so the network must amortize a per-instance non-convex
#   optimization; returns from more stages/samples are now small (runs 6-7:
#   +0.003/+0.004), the signature of an amortization gap.
# New family: run-7 trunk -> learned Kq-aware per-user rate weights alpha
#   (prior = hard-SLqP gradient: 1 for stage in-cut users, 0.1 otherwise; times
#   exp(zero-init head)) -> N_WMMSE=3 unrolled closed-form WMMSE iterations
#   (MMSE receivers, MSE weights, transmit update with per-BS multipliers from
#   a fixed 50-step bisection, detached; exact per-BS rescale). Fixed iteration
#   counts, no utility-based selection, no MQFT: interpreted as the charter's
#   allowed "fixed predeclared number of iterations" (flagged for the director).
#   Verified: equal-weight tail increases sum-rate monotonically on training
#   channels (346 -> 711 -> 778 -> 788 Mbps for 0..3 iterations).
#   BATCH 32 (batch 48 projected ~650 s > cap).
# Hypothesis: the unrolled-WMMSE family beats the global champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.661589301409.
# Launch commit: 7c9cad86c7ddef3084cd202db845194b9ae4a798
# Authoritative HELDOUTSCORE: 1.582954468330 (inference_s 9.950162, train_s 452.75)
# Falsifier outcome: TRIGGERED (-0.078 vs global champion).
# Result: initial family-3 candidate (family-local best, champion_run8.pt);
#   global champion (run 7) untouched.
# Interpretation: worse at every Kq, even Kq=35 (2.199 vs 2.230) although
#   equal-weight WMMSE from the baseline reaches ~2.3 on training channels.
#   The undamped weighted-sum-rate ascent overrides the trunk's balanced
#   allocation: for any fixed alpha it starves low-weight users, while SLqP
#   optima need rate equalization at the cutoff.

# Run 9 - dualbf_wmmse_unrolled (grace attempt 2/5) - learned WMMSE damping
# Hypothesis: the run-8 deficit comes from undamped WSR ascent overriding the
#   trunk's balanced allocation; learned per-BS per-iteration step sizes
#   (V <- (1-eta)V + eta*WMMSE(V), eta = sigmoid(-2 + head(pooled cell tokens)),
#   feasible by convexity) let the network apply WMMSE only where it helps.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.583454468330 (family best + 0.0005).
# Code change: class DualBFWMMSEDamped; wmmse_tail gains optional eta (run-8
#   path verified bit-identical on champion_run8.pt; run-7 likewise).
# Launch commit: 647a52e9938b99758ca5015fd38a4561d605525f
# Authoritative HELDOUTSCORE: 1.652911553189 (inference_s 9.815488, train_s 463.81)
# Falsifier outcome: NOT triggered (+0.069957 over family best).
# Result: family-local COMMIT (champion_run9.pt).
# Interpretation: damping recovers trunk-level behaviour, but run 9 is still
#   BELOW run 6 (1.657250), the identical trunk at identical batch 32 without a
#   tail; S_35 2.2335 vs 2.2291 (+), S_22 1.4361 vs 1.4450 (-). The network
#   learns to mostly bypass WMMSE: weighted-sum-rate ascent does not help this
#   percentile objective beyond what the receive-aware refiners already do.
#
# FAMILY 3 GRACE WINDOW CLOSED EARLY after 2 launched attempts (runs 8-9;
#   sec. 15 allows "at most five"). S_family = 1.652911553189 is not
#   > S_global + 0.0005 = 1.661589301409 -> dualbf_wmmse_unrolled ABANDONED.
#   Rationale for early close: the best family candidate is below its own
#   tail-free trunk at equal budget, so further tail tuning cannot plausibly
#   clear the global bar. Family slots used: 3 of 6. Global champion: run 7.
#   Family-3 classes remain in this file only so runs 8-9 stay reproducible.

# Run 10 - user_transformer_dualbf - interference-pricing stage features
# Hypothesis: refiners see only each user's own rate/rank/cut status but not the
#   marginal-tradeoff information used by interference-pricing algorithms;
#   adding six detached per-user features under the current beams
#   (post-combiner INR, SINR, caused leakage in victims' noise units, cut-
#   weighted interference price, own cut-weighted marginal benefit, log
#   price/benefit) to each refinement stage improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.661589301409.
# Code change: class UserTransformerDualBF3StageRelPrice (stage inputs
#   N_FEAT2 -> N_FEAT2+6) on the run-7 recipe (batch 48, 4000 steps).
# Launch commit: ae175ccfd7503aa4a40ace66aeee93a8e30d32f0
# Authoritative HELDOUTSCORE: 1.660717893966 (inference_s 6.406489, train_s 397.23)
# Falsifier outcome: TRIGGERED (-0.000371 vs global champion).
# Result: REVERT. train.py code portion restored exactly from the run-7
#   champion launch commit 6f8253d (this also drops the abandoned family-3
#   classes; runs 8-10 remain reproducible from their launch commits).
# Interpretation: small redistribution only (S_1 +0.004, mid/high Kq -0.001);
#   the relational refiners already infer the pricing information from rates,
#   ranks and pairwise leakage. Information is not the bottleneck.

# Run 11 - user_transformer_dualbf - width 64 -> 128
# Hypothesis: the champion is capacity-limited (64-wide tokens must encode a
#   Kq-dependent, interference-graph-aware allocation policy over 3 stages);
#   d_model 64->128 and FF 128->256 (4 heads), otherwise run 7, improves it.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.661589301409.
# Code change: D_MODEL 128, FF_DIM 256.
# Launch commit: 39b4da2c3c4df4cd0fcd59f57f98ce408b54b1f3
# Authoritative HELDOUTSCORE: 1.655229580785 (inference_s 7.002180, train_s 456.26)
# Falsifier outcome: TRIGGERED (-0.005860 vs global champion).
# Result: REVERT; code restored from run-7 launch commit 6f8253d.
# Interpretation: wider is uniformly slightly worse (every Kq -0.002..-0.007):
#   not capacity-limited at this sample budget. Together with runs 4/7 (more
#   samples helped) and 10 (more information did not), the binding constraint
#   looks like gradient noise / sample efficiency.

# Run 12 - user_transformer_dualbf - batch 48 -> 64 via speedups
# Pre-launch diagnostic (training-stream channels only, champion_run7): per-
#   sample normalized-SLqP gradient norms differ ~60x across tasks (Kq=1 ~6.5,
#   Kq=9..27 ~1.2-1.6, Kq=35 ~0.11), i.e. minibatch gradient variance is
#   dominated by low-Kq samples.
# Hypothesis: gradient noise is still binding; batch 48->64 (1.33x samples),
#   made affordable by bit-identical refactors (direct-channel SVD reused for
#   the baseline-rate feature, |H|^2 gains computed once) plus fused CPU Adam,
#   improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.661589301409.
# Launch commit: 36cc728f33a9c94399de12a74398671828117303
# Authoritative HELDOUTSCORE: 1.658690723516 (inference_s 4.184763, train_s 399.47)
# Falsifier outcome: TRIGGERED (-0.002399 vs global champion).
# Result: REVERT; code restored from run-7 launch commit 6f8253d.
# Interpretation: more samples no longer help. Because batch/optimizer numerics
#   change the whole training trajectory, the ~+-0.003 differences of runs 6, 7,
#   10, 12 are probably within trajectory-to-trajectory noise; future
#   hypotheses should target larger effects.

# Run 13 - user_transformer_dualbf - per-BS power-constraint multipliers
# Pre-launch diagnostics (training-stream channels only, champion_run7):
#   (i) at Kq=35 the champion is ~a WMMSE fixed point (10 equal-weight WMMSE
#   iterations add only +0.3%), while WSR updates destroy every lower-Kq
#   objective; (ii) Kq conditioning is well calibrated (policy at Kq' = Kq
#   always beats Kq' = Kq +-1,2,4).
# Hypothesis: the beam manifold lacks the per-BS dual of the power constraint:
#   KKT stationarity for max sum_{cut} r under per-BS power gives
#   v_kc ~ (sum_j nu_j h_jc h_jc^H + mu_c I)^-1 h_kc, but the champion fixes
#   mu_c = 1 for all BSs and cannot emulate it through the BS-shared q_j.
#   Learned per-BS log mu_c (residual per stage from the cell's pooled tokens,
#   zero-init) improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.661589301409.
# Code change: class UserTransformerDualBF3StageRelMu (mu heads created after
#   all run-7 parameters, so init/trajectory are paired with run 7; verified
#   zero-init output == champion_run7 exactly).
# Launch commit: 90016afeb25439afa2cf9c9f02abe3fbc8052e48
# Authoritative HELDOUTSCORE: 1.667027600124 (inference_s 5.790308, train_s 373.26)
# Falsifier outcome: NOT triggered (+0.005938 over global champion).
# Result: COMMIT -> new global champion (checkpoint champion_run13.pt).
# Interpretation: gains concentrate at low Kq (S_1 2.189->2.220, S_2
#   1.975->2.006, S_5 1.778->1.796), where a BS must concentrate power on one
#   weak user; mid/high Kq essentially unchanged. Structural completion of the
#   KKT beam form is worth more than budget/capacity/information tweaks.

# Run 14 - user_transformer_dualbf - annealed smooth bottom-Kq selection
# (Pre-launch side diagnostic, training channels only: max-min-style WMMSE
#   heuristics started from the run-13 Kq=1 output end far BELOW it (1.09-1.21
#   vs 2.32 normalized), so they cannot bound the remaining headroom.)
# Hypothesis: the hard SLqP subgradient puts all gradient on exactly the
#   current bottom-Kq set (one user at Kq=1), making the rate-equalizing
#   policies of low/mid-Kq optima slow to learn. An entropic smoothing of the
#   selection, w_i = sigmoid((theta - log r_i)/tau) with sum w = Kq (detached
#   bisection; grad = w by Danskin), annealed tau 0.3 -> 0 over the first 75%
#   of steps and exact hard SLqP for the last 25%, improves the champion.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.667527600124.
# Code change: smooth_cut_weights + surrogate loss -(sum w r)/m_Kq (unit-checked:
#   sum w = Kq, tau=0 equals hard SLqP exactly). Init/data paired with run 13.
# Launch commit: 11f093f89f92b314f78e716ae4f3b6bfd679edf9
# Authoritative HELDOUTSCORE: 1.670968077057 (inference_s 5.911961, train_s 383.71)
# Falsifier outcome: NOT triggered (+0.003940 over global champion).
# Result: COMMIT -> new global champion (checkpoint champion_run14.pt).
# Interpretation: broad gain, largest at mid Kq (S_22 1.4504->1.4570, S_27
#   1.5278->1.5347) where near-cutoff ties matter most; low Kq +0.002..0.005.

# Run 15 - user_transformer_dualbf - deep supervision of intermediate stages
# Hypothesis: stages 1-2 are trained only through the final output, yet their
#   beams determine the MMSE receivers and rate/rank features consumed by later
#   stages; auxiliary smoothed normalized-SLqP losses on the stage-1 and
#   stage-2 beams (weights 0.2, 0.3; rates already computed in the forward)
#   improve the final score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.671468077057.
# Code change: forward_aux returns intermediate stage rates (forward unchanged,
#   verified bit-identical on champion_run14); AUX_WEIGHTS = (0.2, 0.3).
# Launch commit: d5abe44533b7f917e7c1939d440f980535117bcf
# Authoritative HELDOUTSCORE: 1.681217456788 (inference_s 5.803965, train_s 470.09)
# Falsifier outcome: NOT triggered (+0.010249 over global champion).
# Result: COMMIT -> new global champion (checkpoint champion_run15.pt).
# Interpretation: largest gain since run 5, at every Kq (S_1 2.226->2.249,
#   S_9 1.667->1.680, S_22 1.457->1.468; S_35 flat). Well-shaped intermediate
#   beams make the receive-aware refinement far more effective. train_s rose
#   384 -> 470 s (aux backward paths), limiting further depth at this budget.

# Run 16 - user_transformer_dualbf - stronger deep supervision
# Hypothesis: the deep-supervision benefit is not saturated at (0.2, 0.3);
#   aux weights (0.5, 0.7) further improve the final score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.681717456788.
# Code change: AUX_WEIGHTS = (0.5, 0.7).
# Launch commit: 9b247486cc14d6745fe9f4af4fc33d5598bd2e9e
# Authoritative HELDOUTSCORE: 1.681989465329 (inference_s 5.899471, train_s 474.87)
# Falsifier outcome: NOT triggered (+0.000772 over global champion; clears the
#   +0.0005 band by only 0.00027).
# Result: COMMIT -> new global champion (checkpoint champion_run16.pt).
# Interpretation: marginal, slightly larger at low Kq (S_1 2.2488->2.2507);
#   the deep-supervision weight is essentially saturated.

# Run 17 - user_transformer_dualbf - fourth relational stage
# Hypothesis: with deep supervision every stage is a good iterate, so the
#   receive-aware refinement depth is binding again; an untied fourth
#   relational stage (+ its own mu head; aux weights 0.5/0.7/0.85 on stages
#   1-3) improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.682489465329.
# Code change: class UserTransformerDualBF4StageRelMu. Enablers to fit the
#   600 s cap (measured loop x observed real-run factor ~1.43 projected ~625 s
#   at batch 48): bit-identical refactors (direct SVD reused for the
#   baseline-rate feature, |H|^2 gains computed once, one batched row-
#   independent bisection for all smoothed-selection weights; verified on
#   champion_run16 / unit test, max diff 0.0), fused CPU Adam, BATCH 48 -> 40.
# Launch commit: 24a061ba297d4f16076ee02f7ae853785074ce9a
# Authoritative HELDOUTSCORE: 1.686201989926 (inference_s 5.198542, train_s 450.50)
# Falsifier outcome: NOT triggered (+0.004213 over global champion).
# Result: COMMIT -> new global champion (checkpoint champion_run17.pt).
# Interpretation: broad gain (S_1 2.2507->2.2539, S_22 1.4685->1.4735,
#   S_30 1.6628->1.6684); depth pays again under deep supervision, despite
#   batch 48->40.

# Run 18 - user_transformer_dualbf - higher peak LR with warmup
# Hypothesis: LR has stayed at the seed's 1e-3 although the model is now a deep
#   deep-supervised 4-stage net; it is under-optimized within 4000 steps and a
#   peak LR of 2e-3 with a 200-step linear warmup (then cosine to 1e-5)
#   reaches a better solution.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.686701989926.
# Code change: LR 2e-3, WARMUP_STEPS 200, SequentialLR(LinearLR, Cosine).
# Launch commit: 768742e3268aebfbab88e66590759b3accd0bdd6
# Authoritative HELDOUTSCORE: 1.686715818726 (inference_s 5.308236, train_s 455.04)
# Falsifier outcome: NOT triggered, but only by 0.000014 (+0.000514 over the
#   global champion vs the +0.0005 requirement).
# Result: COMMIT per the frozen rule -> new global champion (champion_run18.pt).
# Interpretation: essentially neutral (training logs almost identical to run
#   17; low-Kq +0.001..0.002, Kq>=27 -0.0003); optimization is not
#   LR-limited. Noted as a band-edge commit.

# Run 19 - user_transformer_dualbf - dynamic pairwise interference in biases
# Hypothesis: refiner attention biases use only static channel-gain pairs, but
#   refinement decisions hinge on who currently interferes with whom; adding
#   four detached dynamic pair features (post-MMSE interference from i's stream
#   to user j in j's noise units, its transpose, cut flags of j and i) to each
#   refinement stage's bias MLP improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.687215818726.
# Code change: class UserTransformerDualBF4StageRelMuDyn; training_rates_parts
#   (identical math, also returns stream/noise powers; verified on run 18).
# Launch commit: 6ae712133452b1a926e7d218c2be4fd73738d491
# Authoritative HELDOUTSCORE: 1.687023230074 (inference_s 5.683544, train_s 465.84)
# Falsifier outcome: TRIGGERED (+0.000307 vs global champion, inside band).
# Result: REVERT; code restored from run-18 launch commit 768742e.
# Interpretation: as in run 10, beam-dependent interference information, even
#   routed pairwise, adds nothing the refiners do not already infer from rates,
#   ranks and static coupling.

# Run 20 - user_transformer_dualbf - 5 stages with 1-layer refiners
# Hypothesis: under deep supervision the number of receive-aware refinement
#   iterations matters more than per-stage transformer depth; reallocating
#   compute from 4 stages x 2-layer refiners to 5 stages x 1-layer refiners
#   (stage 1 keeps 2 layers; 5th stage has its own mu head; aux weights
#   0.5/0.7/0.85/0.9) improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.687215818726.
# Code change: class UserTransformerDualBF5StageRelMuL1, AUX_WEIGHTS 4-tuple.
# Launch commit: fd5b36438f17363298f8af0168f2631e348c6b45
# Authoritative HELDOUTSCORE: 1.687124038963 (inference_s 5.813012, train_s 481.76)
# Falsifier outcome: TRIGGERED (+0.000408 vs global champion, inside band).
# Result: REVERT; code restored from run-18 launch commit 768742e.
# Interpretation: refinement depth has saturated at 4 stages (high Kq +0.001,
#   low/mid Kq +-0.0005).

# Run 21 - user_transformer_dualbf - static spatial-alignment pair features
# Hypothesis: the attention bias knows pairwise gains but not pairwise spatial
#   alignment, which sets how cheaply BS(i) can null toward user j; adding
#   log10(M ||H_{j,BS(i)} w_i||^2 / ||H_{j,BS(i)}||_F^2) (w_i = i's unit
#   dominant direct beam; 1 = isotropic) and its transpose to every stage's
#   bias improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.687215818726.
# Code change: class UserTransformerDualBF4StageRelMuSpatial (verified against
#   a manual computation; off-diagonal mean of 10^a = 1.003).
# Launch commit: 227d57c7671f6a81683b5005019c5b8759a97a3e
# Authoritative HELDOUTSCORE: 1.687134335725 (inference_s 6.451376, train_s 475.39)
# Falsifier outcome: TRIGGERED (+0.000419 vs global champion, inside band).
# Result: REVERT; code restored from run-18 launch commit 768742e.
# Interpretation: runs 19-21 (dynamic interference pairs, 5 thin stages,
#   spatial-alignment pairs) all land at 1.6870-1.6871: a plateau of this
#   architecture/training budget; extra information and depth are saturated.

# Run 22 - user_transformer_dualbf - stronger/longer loss smoothing
# Pre-launch headroom diagnostic (TRAINING-stream channels only, 6 channels,
#   scratch script; frozen offline MQFT reference prepare7.mqft_solve_one,
#   never used by train.py or inference): champion_run18 vs MQFT, ratio of
#   means vs baseline: Kq=1 2.176 vs 2.186, Kq=9 1.746 vs 1.755, Kq=18 1.515
#   vs 1.525; per-channel model/MQFT 0.987-0.999. The learned one-shot mapping
#   is within ~0.5-0.65% of the iterative reference: the realistic remaining
#   headroom is small.
# Hypothesis: the residual gap lies in fine allocation among near-tied users
#   around the cutoff, which the smoothed objective addresses (run 14);
#   stronger and longer smoothing (tau0 0.3 -> 0.5, annealed to 0 only at the
#   final step instead of at 75%) closes part of it.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.687215818726.
# Code change: SMOOTH_TAU0 0.5, SMOOTH_END_FRAC 1.0.
# Launch commit: 430dbec9afa59de30d4cc79826fa72173b0db01a
# Authoritative HELDOUTSCORE: 1.681989565760 (inference_s 5.400357, train_s 463.09)
# Falsifier outcome: TRIGGERED (-0.004726 vs global champion).
# Result: REVERT; code restored from run-18 launch commit 768742e.
# Interpretation: worse at every Kq (S_1 -0.011, S_22 -0.003): the run-14
#   benefit needs a final phase on the exact hard objective; heavier smoothing
#   biases the learned allocation.

# Run 23 - user_transformer_dualbf - unbiased importance-sampled Kq law
# Pre-launch diagnostic (12 TRAINING-stream channels, champion_run18): per-
#   sample RMS gradient norm of the normalized final SLqP by Kq: 8.7 (Kq=1),
#   2-3 (Kq=2-9), 1.0-1.8 (Kq=10-33), 0.65 (Kq=34), 0.14 (Kq=35).
# Hypothesis: sampling Kq with p ~ max(RMS, 0.4) and weights 1/(35p) (unbiased
#   for the HELDOUTSCORE-aligned objective; ~40% lower minibatch gradient
#   variance, ~1.7x effective batch) improves the score.
# Falsifier: DIRECTOR_HELDOUTSCORE <= 1.687215818726.
# Code change: KQ_GRAD_RMS constants, multinomial Kq sampling, per-sample
#   importance weights on final and auxiliary losses (checked: E_p[w] = 1).
# Launch commit: 318bf027ec844f02567043d24f86edc8f86f9a75
# Authoritative HELDOUTSCORE: 1.686440753124 (inference_s 5.245702, train_s 453.65)
# Falsifier outcome: TRIGGERED (-0.000275 vs global champion, inside band).
# Result: REVERT; code restored from run-18 launch commit 768742e.
# Interpretation: as with batch 48->64 (run 12), lower gradient variance does
#   not help: the champion is not variance-limited.
#
# =============================================================================
# CAMPAIGN EARLY STOP (sec. 13) after launched run 23 (23 of 50 used)
# =============================================================================
# Why further search is unlikely to help:
#   * Five consecutive non-commits (runs 19-23): runs 19, 20, 21, 23 land within
#     +-0.0005 of the champion (1.68644-1.68713) and run 22 is clearly worse;
#     the champion itself (run 18) was a band-edge commit over run 17.
#   * Headroom diagnostic on TRAINING-stream channels against the frozen
#     offline MQFT reference (scratch script only): the champion reaches
#     99.4-99.5% of MQFT's ratio-of-means at Kq = 1, 9, 18 (per-channel
#     0.987-0.999). The remaining realistic headroom (~0.5-0.65%) is of the
#     order of trajectory-to-trajectory noise for single runs.
#   * Continuing with neutral variants would mainly convert selection-set
#     noise into band-edge commits (winner's curse on 250 channels) rather
#     than real improvements.
# Major directions tested:
#   * Architecture families (3 of 6 opened): centralized_mlp (seed; runs 0-1),
#     user_transformer_dualbf (runs 2-7, 10-23; promoted after its grace
#     window), dualbf_wmmse_unrolled (runs 8-9; closed early, abandoned).
#   * Output parameterization / physics structure: optimal dual-power beam
#     structure (2), receive-aware MMSE refinement stages (3, 6, 17, 20),
#     per-BS power-constraint multipliers mu_c (13), unrolled WMMSE (8, 9).
#   * Features / relational information: relational attention bias (5),
#     interference-pricing features (10), dynamic pairwise interference (19),
#     spatial-alignment pairs (21); Kq calibration diagnostic (well calibrated).
#   * Objective: score-aligned per-Kq normalization (1, 2), annealed entropic
#     smoothing of the bottom-Kq selection (14, 22), deep supervision (15, 16).
#   * Optimization/budget: steps/batch/cosine LR (4, 7, 12), width (11),
#     peak LR + warmup (18), importance-sampled Kq law (23).
# Stopping run ID: 23.
# FROZEN GLOBAL CHAMPION: run 18, family user_transformer_dualbf,
#   class UserTransformerDualBF4StageRelMu, DIRECTOR_HELDOUTSCORE
#   1.686715818726, launch commit 768742e3268aebfbab88e66590759b3accd0bdd6,
#   checkpoint champion_run18.pt (sha256 ae5189ea6ad9deb5eb5dcc057f9d2ff917c3
#   9962b82cb9eef2061b4ffd8f6043). The code portion of this file equals that
#   launch commit's. The protected final evaluation (director-held seed and
#   lock) has NOT been run and is left to the director.
