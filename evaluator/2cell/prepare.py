"""
prepare.py -- Baby Percentile Beamforming v1.1 evaluator.

PRE-CAMPAIGN STATUS
-------------------
This file implements the frozen Baby-v1.1 physics/evaluation decisions made
before the autonomous campaign. Calibration and the numerical inference budget
are finalized; this file becomes immutable and SHA-256-pinned before the
autonomous campaign begins.

Model API
---------
    V = model(H, Kq)

    H : complex tensor [batch, 4, 2, 2, 2, 8]
        H[t,k,b,bp,:,:] = channel from BS bp to user k in cell b.
    Kq: int or integer tensor [batch], values in {1,...,8}.
    V : complex tensor [batch, 4, 2, 8]
        V[t,k,b,:] = transmit beamformer for user k in cell b.

The authoritative selection metric is HELDOUTSCORE: for each Kq, the model's
mean SLqP on 250 pinned channels divided by the mean SLqP of the equal-power
dominant-direct-channel-singular-vector baseline on the SAME channels, then
averaged equally over Kq=1,...,8.

The final set is protocol-protected: it is generated lazily and must not be
queried during the research campaign. evaluate_final(...) therefore requires
an explicit authorization flag and should only be used after the champion is
frozen.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import os

# =============================================================================
# Frozen Baby Percentile Beamforming v1.1 constants
# =============================================================================
B = 2                     # cells
K_USERS = 4               # users per cell
K_TOTAL = B * K_USERS     # 8 total users
M_TX = 8                  # BS transmit antennas
N_RX = 2                  # user receive antennas

APOTHEM_M = 1000.0
BS_SPACING_M = 2000.0
HEX_CIRCUMRADIUS_M = 2.0 * APOTHEM_M / math.sqrt(3.0)

# Exact coordinates/orientation chosen to match the Power-Control Autoresearch
# hexagon convention: 30-degree-offset vertices, adjacent BSs on the x-axis.
BS_POS = np.array([0.0 + 0.0j, BS_SPACING_M + 0.0j], dtype=np.complex128)

D0_M = 0.392
PATHLOSS_EXPONENT = 3.76

P_MAX_DBM = 43.0
P_MAX_W = 10.0 ** ((P_MAX_DBM - 30.0) / 10.0)
SQRT_P_MAX_W = math.sqrt(P_MAX_W)
BASELINE_STREAM_POWER_W = P_MAX_W / K_USERS
SQRT_BASELINE_STREAM_POWER_W = math.sqrt(BASELINE_STREAM_POWER_W)

NOISE_PSD_DBM_HZ = -143.0
BANDWIDTH_HZ = 20e6
NOISE_POWER_W = 10.0 ** ((NOISE_PSD_DBM_HZ - 30.0) / 10.0) * BANDWIDTH_HZ

SEED_SELECT = 123
SEED_TRAIN = 789
TORCH_SEED = 2027


def _final_seed() -> int:
    s = os.environ.get("BABY_FINAL_SEED")
    if s is None:
        raise PermissionError("Final seed is held by the director until champion freeze.")
    return int(s)

N_SELECT = 250
N_FINAL = 250
KQ_VALUES = tuple(range(1, K_TOTAL + 1))

# Evaluator precision is frozen at complex128 / float64.
EVAL_COMPLEX_DTYPE = torch.complex128
EVAL_REAL_DTYPE = torch.float64

# Beamformer validation tolerance: numerical guard only, not extra usable power.
POWER_REL_TOL = 1e-6

# Fixed inference budget over the whole selection grid (8 Kq x 250 drops).
# Scaled from the power-control campaign (10 s / 4250 tasks ~ 2.4 ms/task).
INF_BUDGET_S: Optional[float] = 5.0

# Five-repeat score-noise calibration is complete. The resulting delta is recorded
# in program.md rather than here because it controls outer-loop commit/revert,
# not physical evaluation.

# =============================================================================
# Geometry + channel generation (immutable physics)
# =============================================================================
def _drop_users_hex(n: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly uniform drops in a regular hexagon via six equal-area triangles.

    This follows the sampling method used in the Power-Control Autoresearch:
    choose one of six triangles uniformly, then use reflected barycentric draws.
    Returns complex offsets from the serving BS center.
    """
    tri = rng.integers(0, 6, n)
    a0 = np.deg2rad(30.0 + 60.0 * tri)
    a1 = np.deg2rad(30.0 + 60.0 * (tri + 1))
    v0 = HEX_CIRCUMRADIUS_M * np.exp(1j * a0)
    v1 = HEX_CIRCUMRADIUS_M * np.exp(1j * a1)

    u1 = rng.random(n)
    u2 = rng.random(n)
    flip = (u1 + u2) > 1.0
    u1[flip] = 1.0 - u1[flip]
    u2[flip] = 1.0 - u2[flip]
    return u1 * v0 + u2 * v1


def _sample_channels_from_rng(
    batch: int,
    rng: np.random.Generator,
    *,
    dtype: torch.dtype = EVAL_COMPLEX_DTYPE,
) -> torch.Tensor:
    """Generate Baby-v1.1 channels from ONE sequential RNG stream.

    Frozen RNG draw order:
      1) user positions,
      2) all real Rayleigh Gaussian draws,
      3) all imaginary Rayleigh Gaussian draws.
    """
    if batch <= 0:
        raise ValueError("batch must be positive")

    # [batch, B, K_USERS] complex offsets, then absolute user positions.
    offsets = _drop_users_hex(batch * B * K_USERS, rng).reshape(batch, B, K_USERS)
    user_pos = BS_POS[None, :, None] + offsets                        # [T,B,K]
    user_pos = np.transpose(user_pos, (0, 2, 1))                     # [T,K,B]

    # Distance from every receiving user to each transmitting BS.
    dist = np.abs(user_pos[:, :, :, None] - BS_POS[None, None, None, :])
                                                                    # [T,K,B,Bp]
    path_amp = (1.0 + dist / D0_M) ** (-PATHLOSS_EXPONENT / 2.0)

    shape = (batch, K_USERS, B, B, N_RX, M_TX)
    real = rng.normal(0.0, 1.0, size=shape)
    imag = rng.normal(0.0, 1.0, size=shape)
    fading = (real + 1j * imag) / math.sqrt(2.0)                    # CN(0,1)

    H = path_amp[..., None, None] * fading
    np_dtype = np.complex128 if dtype == torch.complex128 else np.complex64
    return torch.from_numpy(H.astype(np_dtype, copy=False))


def sample_channels(batch: int, seed: int, *, dtype: torch.dtype = EVAL_COMPLEX_DTYPE) -> torch.Tensor:
    """Deterministic one-shot channel set for a supplied seed."""
    rng = np.random.default_rng(seed)
    return _sample_channels_from_rng(batch, rng, dtype=dtype)


class TrainingChannelStream:
    """Fresh Baby-v1.1 training channels from the fixed SEED_TRAIN stream.

    Construct a new instance at the start of every complete experiment. Doing so
    resets the training-channel sequence exactly, as frozen in C2.30.
    """

    def __init__(self, seed: int = SEED_TRAIN) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self, batch: int, *, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
        return _sample_channels_from_rng(batch, self._rng, dtype=dtype)


# Pinned selection channels. The final channels are deliberately NOT materialized
# during normal campaign operation.
_SELECT_H = sample_channels(N_SELECT, SEED_SELECT, dtype=EVAL_COMPLEX_DTYPE)

# =============================================================================
# Exact evaluator physics
# =============================================================================
def _check_H(H: torch.Tensor) -> torch.Tensor:
    expected_tail = (K_USERS, B, B, N_RX, M_TX)
    if H.ndim != 6 or tuple(H.shape[1:]) != expected_tail:
        raise ValueError(f"H shape {tuple(H.shape)} != [batch,{','.join(map(str, expected_tail))}]")
    if not torch.is_complex(H):
        raise ValueError("H must be a complex tensor")
    return H.to(device="cpu", dtype=EVAL_COMPLEX_DTYPE)


def _received_stream_vectors(H: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Return received vectors for every receiver/stream pair.

    Output S[t,k,b,j,c,n] = H[t,k,b,c,:,:] @ V[t,j,c,:], where
    (k,b) is the receiving user and (j,c) is the transmitted stream.
    """
    return torch.einsum("tkbcnm,tjcm->tkbjcn", H, V)


def mmse_rates_mbps(H: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Authoritative rates using analytical MMSE receive combiners.

    Returns
    -------
    rates : [batch, K_USERS, B] in Mbps
    u     : [batch, K_USERS, B, N_RX] complex MMSE combiners

    u = R_yy^{-1} H_serv v_des, where R_yy contains noise + ALL streams.
    SINR is then evaluated with the desired stream excluded from the denominator.
    """
    H = _check_H(H)
    V = V.to(device="cpu", dtype=EVAL_COMPLEX_DTYPE)
    batch = H.shape[0]

    S = _received_stream_vectors(H, V)                              # [T,K,B,K,B,N]

    # Desired vector for each receiving user.
    desired = torch.empty((batch, K_USERS, B, N_RX), dtype=EVAL_COMPLEX_DTYPE)
    for k in range(K_USERS):
        for b in range(B):
            desired[:, k, b, :] = S[:, k, b, k, b, :]

    eye = torch.eye(N_RX, dtype=EVAL_COMPLEX_DTYPE).view(1, 1, 1, N_RX, N_RX)
    Ryy = torch.einsum("tkbjcn,tkbjcm->tkbnm", S, S.conj())
    Ryy = Ryy + NOISE_POWER_W * eye

    # Frozen numerical implementation: direct linear solve, no artificial jitter.
    u = torch.linalg.solve(Ryy, desired.unsqueeze(-1)).squeeze(-1)

    # Project every transmitted stream onto the MMSE receive combiner.
    proj = torch.einsum("tkbn,tkbjcn->tkbjc", u.conj(), S)
    stream_power = proj.abs().square()

    signal = torch.empty((batch, K_USERS, B), dtype=EVAL_REAL_DTYPE)
    interference = stream_power.sum(dim=(-1, -2))
    for k in range(K_USERS):
        for b in range(B):
            sig = stream_power[:, k, b, k, b]
            signal[:, k, b] = sig
            interference[:, k, b] = interference[:, k, b] - sig

    noise_out = NOISE_POWER_W * u.abs().square().sum(dim=-1)
    sinr = signal / (interference + noise_out)
    rates = torch.log2(1.0 + sinr) * (BANDWIDTH_HZ / 1e6)
    return rates.real, u


def slqp_from_rates(rates_mbps: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
    """Exact network-wide sum of the Kq smallest rates.

    Supports either one integer Kq for the whole batch or a [batch] tensor of
    per-sample Kq values.
    """
    if rates_mbps.ndim == 3:
        flat = rates_mbps.reshape(rates_mbps.shape[0], K_TOTAL)
    elif rates_mbps.ndim == 2 and rates_mbps.shape[1] == K_TOTAL:
        flat = rates_mbps
    else:
        raise ValueError(f"rates shape {tuple(rates_mbps.shape)} is incompatible with K_TOTAL={K_TOTAL}")

    sorted_rates = torch.sort(flat, dim=-1).values
    prefix = torch.cumsum(sorted_rates, dim=-1)

    if isinstance(Kq, int):
        if not 1 <= Kq <= K_TOTAL:
            raise ValueError(f"Kq={Kq} outside 1..{K_TOTAL}")
        return prefix[:, Kq - 1]

    kq = torch.as_tensor(Kq, device=prefix.device, dtype=torch.long)
    if kq.ndim == 0:
        kq = kq.expand(prefix.shape[0])
    if tuple(kq.shape) != (prefix.shape[0],):
        raise ValueError(f"Kq tensor shape {tuple(kq.shape)} != ({prefix.shape[0]},)")
    if bool(((kq < 1) | (kq > K_TOTAL)).any()):
        raise ValueError(f"all Kq values must lie in 1..{K_TOTAL}")
    return prefix.gather(1, (kq - 1).unsqueeze(1)).squeeze(1)


def evaluate_slqp(H: torch.Tensor, V: torch.Tensor, Kq: int | torch.Tensor) -> torch.Tensor:
    """Exact immutable Baby-v1.1 SLqP values in Mbps."""
    rates, _ = mmse_rates_mbps(H, V)
    return slqp_from_rates(rates, Kq)


# =============================================================================
# Equal-power dominant-direct-channel-SVD normalization baseline
# =============================================================================
def equal_power_svd_beamformers(H: torch.Tensor) -> torch.Tensor:
    """Fixed baseline: Pmax/K_USERS per stream along dominant right singular vectors."""
    H = _check_H(H)
    direct = torch.stack([H[:, :, b, b, :, :] for b in range(B)], dim=2)
                                                                    # [T,K,B,N,M]
    _, _, Vh = torch.linalg.svd(direct, full_matrices=False)
    vmax = Vh[..., 0, :].conj()                                     # right singular vector
    return SQRT_BASELINE_STREAM_POWER_W * vmax


def _baseline_means(H: torch.Tensor) -> Dict[int, float]:
    with torch.no_grad():
        Vb = equal_power_svd_beamformers(H)
        rates, _ = mmse_rates_mbps(H, Vb)
        return {kq: float(slqp_from_rates(rates, kq).mean().item()) for kq in KQ_VALUES}


_BASELINE_SELECT_MEAN = _baseline_means(_SELECT_H)


# =============================================================================
# Model validation + selection/final evaluation
# =============================================================================
def _validate_model_output(V: torch.Tensor, batch: int) -> Dict[str, float]:
    expected = (batch, K_USERS, B, M_TX)
    if tuple(V.shape) != expected:
        raise RuntimeError(f"FAIL: model output shape {tuple(V.shape)} != {expected}")
    if not torch.is_complex(V):
        raise RuntimeError("FAIL: model output V must be complex")
    if not bool(torch.isfinite(V.real).all() and torch.isfinite(V.imag).all()):
        raise RuntimeError("FAIL: model output contains NaN or Inf")

    V128 = V.to(device="cpu", dtype=EVAL_COMPLEX_DTYPE)
    user_power = V128.abs().square().sum(dim=-1)                  # [batch,K,B]
    bs_power = user_power.sum(dim=1)                              # [batch,B]
    max_bs_power = float(bs_power.max().item())
    max_user_power = float(user_power.max().item())
    allowed = P_MAX_W * (1.0 + POWER_REL_TOL)
    if bool((bs_power > allowed).any()):
        raise RuntimeError(
            f"FAIL: per-BS total-power violation: max sum_k ||v_k,b||^2="
            f"{max_bs_power:.12g} W > Pmax*(1+tol)={allowed:.12g} W"
        )
    return {
        "max_bs_power_w": max_bs_power,
        "max_user_power_w": max_user_power,
    }


def _snapshot_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _assert_state_unchanged(model: torch.nn.Module, before: Dict[str, torch.Tensor]) -> None:
    after = model.state_dict()
    if set(after) != set(before):
        raise RuntimeError("FAIL: inference contract violated: model state_dict keys changed")
    for k, v0 in before.items():
        v1 = after[k].detach().cpu()
        if not torch.equal(v0, v1):
            raise RuntimeError(f"FAIL: inference contract violated: model state changed at '{k}'")


def _model_kq(batch: int, kq: int) -> torch.Tensor:
    return torch.full((batch,), kq, dtype=torch.long)


def _run_grid_once(
    model: torch.nn.Module,
    H: torch.Tensor,
    baseline_means: Dict[int, float],
    *,
    enforce_budget: bool,
    verbose: bool,
) -> Dict[str, object]:
    model.eval()
    state_before = _snapshot_state(model)

    # Unmeasured warm-up, as frozen in the inference calibration method.
    with torch.no_grad():
        _ = model(H[: min(8, H.shape[0])].clone(), _model_kq(min(8, H.shape[0]), 1))
    _assert_state_unchanged(model, state_before)

    score_by_kq: Dict[int, float] = {}
    raw_slqp_mbps_by_kq: Dict[int, float] = {}
    inference_time_s = 0.0
    validity = {"max_bs_power_w": 0.0, "max_user_power_w": 0.0}

    for kq in KQ_VALUES:
        kq_input = _model_kq(H.shape[0], kq)
        H_in = H.clone()
        t0 = time.perf_counter()
        with torch.no_grad():
            V = model(H_in, kq_input)
        inference_time_s += time.perf_counter() - t0

        vd = _validate_model_output(V, H.shape[0])
        validity["max_bs_power_w"] = max(validity["max_bs_power_w"], vd["max_bs_power_w"])
        validity["max_user_power_w"] = max(validity["max_user_power_w"], vd["max_user_power_w"])

        with torch.no_grad():
            val = float(evaluate_slqp(H, V, kq).mean().item())
        raw_slqp_mbps_by_kq[kq] = val
        score_by_kq[kq] = val / baseline_means[kq]

    _assert_state_unchanged(model, state_before)

    if enforce_budget and INF_BUDGET_S is not None and inference_time_s > INF_BUDGET_S:
        raise RuntimeError(
            f"FAIL: inference budget violated: {inference_time_s:.6f}s > {INF_BUDGET_S:.6f}s "
            f"for all {N_SELECT * len(KQ_VALUES)} selection tasks"
        )

    heldoutscore = float(np.mean([score_by_kq[kq] for kq in KQ_VALUES]))
    result: Dict[str, object] = {
        "HELDOUTSCORE": heldoutscore,
        "score_by_kq": score_by_kq,
        "raw_slqp_mbps_by_kq": raw_slqp_mbps_by_kq,
        "baseline_slqp_mbps_by_kq": dict(baseline_means),
        "inference_time_s": inference_time_s,
        "inference_budget_s": INF_BUDGET_S,
        "inference_budget_pass": (INF_BUDGET_S is None or inference_time_s <= INF_BUDGET_S),
        "validity": validity,
    }

    if verbose:
        print(f"INFERENCE_S {inference_time_s:.6f}")
        for kq in KQ_VALUES:
            print(
                f"KQ {kq}  SCORE {score_by_kq[kq]:.6f}  "
                f"RAW_SLQP_MBPS {raw_slqp_mbps_by_kq[kq]:.6f}  "
                f"BASE_MBPS {baseline_means[kq]:.6f}"
            )
        print(f"HELDOUTSCORE {heldoutscore:.6f}")
        if INF_BUDGET_S is None:
            print("INFERENCE_BUDGET CALIBRATION_PENDING")

    return result


def evaluate_selection(model: torch.nn.Module, *, verbose: bool = True) -> Dict[str, object]:
    """Authoritative repeated campaign evaluation on D_select."""
    return _run_grid_once(
        model,
        _SELECT_H,
        _BASELINE_SELECT_MEAN,
        enforce_budget=True,
        verbose=verbose,
    )


def measure_inference_time(
    model: torch.nn.Module,
    *,
    repeats: int,
    warmups: int,
) -> Tuple[float, Tuple[float, ...]]:
    """Calibration helper: median (H,Kq)->V time over the complete selection grid.

    Timing excludes channel generation, scoring, file I/O, and process startup.
    The caller MUST explicitly choose repeats/warmups; those counts were not yet
    frozen when this pre-campaign file was generated.
    """
    if repeats <= 0 or warmups < 0:
        raise ValueError("repeats must be >0 and warmups must be >=0")

    model.eval()
    state_before = _snapshot_state(model)
    with torch.no_grad():
        for _ in range(warmups):
            for kq in KQ_VALUES:
                _ = model(_SELECT_H.clone(), _model_kq(N_SELECT, kq))

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            H_ins = [_SELECT_H.clone() for _ in KQ_VALUES]
            t0 = time.perf_counter()
            for H_in, kq in zip(H_ins, KQ_VALUES):
                _ = model(H_in, _model_kq(N_SELECT, kq))
            times.append(time.perf_counter() - t0)

    _assert_state_unchanged(model, state_before)
    return float(np.median(times)), tuple(float(x) for x in times)


def evaluate_final(
    model: torch.nn.Module,
    *,
    authorization: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """One-time final evaluation after champion freeze.

    PROTOCOL RULE: do not call this during the Autoresearch campaign. The final
    set is disjoint and independently seeded. No retuning is permitted after its
    result is revealed.
    """
    if not authorization:
        raise PermissionError(
            "Final evaluation is protocol-protected. Set authorization=True only "
            "after the campaign champion is frozen."
        )
    H_final = sample_channels(N_FINAL, _final_seed(), dtype=EVAL_COMPLEX_DTYPE)
    base_final = _baseline_means(H_final)
    return _run_grid_once(
        model,
        H_final,
        base_final,
        enforce_budget=True,
        verbose=verbose,
    )


# =============================================================================
# Offline strong classical reference: MQFT (optional CVXPY dependency)
# =============================================================================
@dataclass
class MQFTResult:
    V: torch.Tensor
    trajectory_mbps: Tuple[float, ...]
    converged: bool
    iterations: int
    n_decreases: int = 0


def _mqft_auxiliaries(H_one: torch.Tensor, V_one: torch.Tensor) -> np.ndarray:
    """Closed-form MQFT chi updates for one channel realization.

    chi_{k,b} = B^{-1} H_serv v_des, where B contains receiver noise and all
    INTERFERING streams (desired stream excluded), matching the MQFT transform.
    """
    H = H_one.to(dtype=EVAL_COMPLEX_DTYPE)
    V = V_one.to(dtype=EVAL_COMPLEX_DTYPE)
    S = _received_stream_vectors(H.unsqueeze(0), V.unsqueeze(0))[0]
    chis = np.empty((K_USERS, B, N_RX), dtype=np.complex128)
    eye = torch.eye(N_RX, dtype=EVAL_COMPLEX_DTYPE)

    for k in range(K_USERS):
        for b in range(B):
            Bmat = NOISE_POWER_W * eye.clone()
            for j in range(K_USERS):
                for c in range(B):
                    if j == k and c == b:
                        continue
                    s = S[k, b, j, c]
                    Bmat = Bmat + s[:, None] @ s.conj()[None, :]
            a = S[k, b, k, b]
            chi = torch.linalg.solve(Bmat, a)
            chis[k, b] = chi.cpu().numpy()
    return chis


def _mqft_v_update_cvxpy(H_one: torch.Tensor, chis: np.ndarray, Kq: int) -> torch.Tensor:
    """Solve the fixed-X concave MQFT V-subproblem under per-BS total-power constraints."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Offline MQFT reference requires cvxpy (with CLARABEL or SCS). "
            "It is not needed for seed training or model inference."
        ) from exc

    Hn = H_one.detach().cpu().numpy().astype(np.complex128, copy=False)
    Vvar = cp.Variable((K_TOTAL, M_TX), complex=True)
    rates = []
    constraints = []

    def sidx(k: int, b: int) -> int:
        return k * B + b

    # Part II Problem (3b): one shared total-power budget per BS.
    for b in range(B):
        rows = [sidx(k, b) for k in range(K_USERS)]
        constraints.append(cp.sum_squares(cp.abs(Vvar[rows, :])) <= P_MAX_W)

    for k in range(K_USERS):
        for b in range(B):
            chi = chis[k, b]
            des = sidx(k, b)
            hdes = Hn[k, b, b]                                      # [N,M]
            coeff_des = chi.conj() @ hdes                            # [M]

            q = 1.0 - NOISE_POWER_W * float(np.vdot(chi, chi).real)
            q = q + 2.0 * cp.real(coeff_des @ Vvar[des, :])

            for j in range(K_USERS):
                for c in range(B):
                    if j == k and c == b:
                        continue
                    coeff = chi.conj() @ Hn[k, b, c]
                    q = q - cp.square(cp.abs(coeff @ Vvar[sidx(j, c), :]))

            # Explicit domain guard makes solver failures easier to diagnose.
            constraints.append(q >= 1e-12)
            rates.append(cp.log(q))                                  # natural-log scale; argmax unchanged

    rvec = cp.hstack(rates)
    obj = cp.sum(rvec) if Kq == K_TOTAL else cp.sum_smallest(rvec, Kq)
    prob = cp.Problem(cp.Maximize(obj), constraints)

    solved = False
    for solver in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and Vvar.value is not None:
                solved = True
                break
        except Exception:
            continue
    if not solved:
        raise RuntimeError(f"MQFT fixed-X V update failed; CVXPY status={prob.status}")

    Vf = np.asarray(Vvar.value, dtype=np.complex128).reshape(K_USERS, B, M_TX)
    bs_power = np.sum(np.abs(Vf) ** 2, axis=(0, 2))
    if np.max(bs_power) > P_MAX_W * (1.0 + 1e-6):
        raise RuntimeError(
            "MQFT solver returned materially infeasible point: "
            f"max BS total power={np.max(bs_power)} W"
        )
    return torch.from_numpy(Vf)


def mqft_solve_one(
    H_one: torch.Tensor,
    Kq: int,
    *,
    rel_tol: float = 1e-6,
    max_iter: int = 25,
) -> MQFTResult:
    """Baby-v1.1 MQFT reference for one channel and one Kq.

    Frozen choices:
      * initialization = equal-power dominant-direct-channel SVD baseline,
      * relative objective tolerance = 1e-6,
      * max iterations = 25,
      * reaching the cap before tolerance is explicitly NOT CONVERGED.
    """
    if not 1 <= Kq <= K_TOTAL:
        raise ValueError(f"Kq={Kq} outside 1..{K_TOTAL}")
    H_one = _check_H(H_one.unsqueeze(0))[0]
    V = equal_power_svd_beamformers(H_one.unsqueeze(0))[0]
    f_prev = float(evaluate_slqp(H_one.unsqueeze(0), V.unsqueeze(0), Kq).item())
    trajectory = [f_prev]
    converged = False
    n_decreases = 0

    for it in range(1, max_iter + 1):
        chis = _mqft_auxiliaries(H_one, V)
        V = _mqft_v_update_cvxpy(H_one, chis, Kq)
        f_new = float(evaluate_slqp(H_one.unsqueeze(0), V.unsqueeze(0), Kq).item())
        if f_new < f_prev * (1.0 - 1e-9):
            n_decreases += 1
        trajectory.append(f_new)
        rel = abs(f_new - f_prev) / max(abs(f_prev), 1e-12)
        if rel < rel_tol:
            converged = True
            return MQFTResult(V=V, trajectory_mbps=tuple(trajectory), converged=True, iterations=it, n_decreases=n_decreases)
        f_prev = f_new

    return MQFTResult(V=V, trajectory_mbps=tuple(trajectory), converged=converged, iterations=max_iter, n_decreases=n_decreases)


def run_mqft_reference(
    *,
    dataset: str = "selection",
    authorization: bool = False,
    cache_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Offline MQFT bar on all channels x all Kq; potentially expensive.

    This routine is NEVER called by model inference. For dataset='final', explicit
    authorization is required and should only be granted after champion freeze.
    """
    if dataset == "selection":
        Hset = _SELECT_H
    elif dataset == "final":
        if not authorization:
            raise PermissionError("Final MQFT reference is protocol-protected until champion freeze")
        Hset = sample_channels(N_FINAL, _final_seed(), dtype=EVAL_COMPLEX_DTYPE)
    else:
        raise ValueError("dataset must be 'selection' or 'final'")

    values = np.empty((Hset.shape[0], K_TOTAL), dtype=np.float64)
    converged = np.zeros((Hset.shape[0], K_TOTAL), dtype=np.bool_)
    iterations = np.zeros((Hset.shape[0], K_TOTAL), dtype=np.int64)
    decreases = np.zeros((Hset.shape[0], K_TOTAL), dtype=np.int64)

    for n in range(Hset.shape[0]):
        for col, kq in enumerate(KQ_VALUES):
            res = mqft_solve_one(Hset[n], kq)
            values[n, col] = res.trajectory_mbps[-1]
            converged[n, col] = res.converged
            iterations[n, col] = res.iterations
            decreases[n, col] = res.n_decreases
            print(
                f"MQFT dataset={dataset} drop={n+1}/{Hset.shape[0]} Kq={kq} "
                f"SLqP={values[n,col]:.6f} converged={bool(converged[n,col])} "
                f"iters={iterations[n,col]}"
            )

    out = {
        "mean_slqp_mbps_by_kq": {kq: float(values[:, i].mean()) for i, kq in enumerate(KQ_VALUES)},
        "converged_fraction_by_kq": {kq: float(converged[:, i].mean()) for i, kq in enumerate(KQ_VALUES)},
        "cap_hits_by_kq": {kq: int((~converged[:, i]).sum()) for i, kq in enumerate(KQ_VALUES)},
        "decreases_by_kq": {kq: int(decreases[:, i].sum()) for i, kq in enumerate(KQ_VALUES)},
        "decreases": decreases,
        "values": values,
        "converged": converged,
        "iterations": iterations,
    }

    if cache_path is not None:
        cache_path = Path(cache_path)
        np.savez_compressed(
            cache_path,
            values=values,
            converged=converged,
            iterations=iterations,
            dataset=dataset,
            decreases=decreases,
        )
    return out


# =============================================================================
# Lightweight self-audit (does not train a model)
# =============================================================================
def self_audit() -> None:
    assert np.allclose(BS_POS, np.array([0.0 + 0.0j, 2000.0 + 0.0j]))
    assert abs(P_MAX_W - 10.0 ** 1.3) < 1e-12
    assert _SELECT_H.shape == (N_SELECT, K_USERS, B, B, N_RX, M_TX)
    assert _SELECT_H.dtype == EVAL_COMPLEX_DTYPE

    with torch.no_grad():
        Vb = equal_power_svd_beamformers(_SELECT_H[:3])
        _validate_model_output(Vb, 3)
        rates, u = mmse_rates_mbps(_SELECT_H[:3], Vb)
        assert rates.shape == (3, K_USERS, B)
        assert u.shape == (3, K_USERS, B, N_RX)
        assert bool(torch.isfinite(rates).all())
        for kq in KQ_VALUES:
            f = slqp_from_rates(rates, kq)
            assert f.shape == (3,)
            assert bool(torch.isfinite(f).all())
    print("prepare.py self_audit: PASS")


if __name__ == "__main__":
    self_audit()
