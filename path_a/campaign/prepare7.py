"""
prepare7.py -- IMMUTABLE evaluator for Direct 7-cell Percentile Beamforming Autoresearch.

Frozen model API
----------------
    V = model(H, Kq)

    H : complex tensor [batch, 5, 7, 7, 2, 8]
        H[t,k,b,bp,:,:] = channel from BS bp to user k in serving cell b.
    Kq: int or integer tensor [batch], values in {1,...,35}.
    V : complex tensor [batch, 5, 7, 8]
        V[t,k,b,:] = transmit beamformer for user k in cell b.

The authoritative selection metric is HELDOUTSCORE. For each Kq=1,...,35,
we divide the candidate's mean network-wide hard-SLqP throughput on the 250
pinned selection channels by the mean of the equal-power dominant-direct-
channel-SVD baseline on the SAME channels. HELDOUTSCORE is the equal-weight
mean of those 35 normalized scores.

The final set is protocol-protected and generated lazily from a director-held
seed supplied only after the campaign champion is frozen.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# =============================================================================
# Frozen 7-cell constants
# =============================================================================
B = 7
K_USERS = 5
K_TOTAL = B * K_USERS
M_TX = 8
N_RX = 2

# Geometry convention inherited from the validated 7-cell torus implementation:
# R is the hexagon apothem; adjacent BS spacing is 2R = 2000 m.
APOTHEM_M = 1000.0
BS_SPACING_M = 2000.0
HEX_CIRCUMRADIUS_M = 2.0 * APOTHEM_M / math.sqrt(3.0)

_th = 1j * np.pi / 3.0
BS_POS = 2.0 * APOTHEM_M * np.array(
    [
        0.0,
        1.0,
        np.exp(_th),
        np.exp(2.0 * _th),
        np.exp(3.0 * _th),
        np.exp(4.0 * _th),
        np.exp(5.0 * _th),
    ],
    dtype=np.complex128,
)


def _rot(z: complex, k: int) -> complex:
    return z * np.exp(1j * np.pi / 3.0 * k)


_t1 = 2.0 * APOTHEM_M * (2.0 + np.exp(_th))
_t2 = _rot(_t1, 1)
SHIFTS = np.array(
    [0.0, _t1, -_t1, _t2, -_t2, _t1 - _t2, _t2 - _t1],
    dtype=np.complex128,
)
assert np.allclose(
    np.abs(SHIFTS[1:]), 2.0 * APOTHEM_M * np.sqrt(7.0)
), "wrap-around shifts must have magnitude 2R*sqrt(7)"

# Cell-level nearest periodic image: one image is chosen per serving-cell/logical-BS
# pair from center-to-center distances, then reused for every user in that cell.
_d_center = np.abs(
    BS_POS[:, None] - (BS_POS[None, :] + SHIFTS[:, None, None])
)
_pick = np.argmin(_d_center, axis=0)
CELL_IMAGE = BS_POS[None, :] + SHIFTS[_pick]  # [serving_cell, logical_bs]

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
N_SELECT = 250
N_FINAL = 250
KQ_VALUES = tuple(range(1, K_TOTAL + 1))
MQFT_KQ_VALUES = (1, 2, 4, 9, 18, 35)

# Evaluator arithmetic is frozen at complex128 / float64.
EVAL_COMPLEX_DTYPE = torch.complex128
EVAL_REAL_DTYPE = torch.float64
POWER_REL_TOL = 1e-6

# Fixed mapping budget for all 250 x 35 selection tasks.
INF_BUDGET_S: Optional[float] = 25.0


def _final_seed() -> int:
    s = os.environ.get("SEVEN_CELL_FINAL_SEED")
    if s is None:
        raise PermissionError(
            "Final seed is held by the director until the champion is frozen."
        )
    seed = int(s)
    if seed in {SEED_SELECT, SEED_TRAIN}:
        raise RuntimeError(
            "FAIL: final seed must be independent of selection/training seeds "
            f"({SEED_SELECT}, {SEED_TRAIN}); got {seed}"
        )
    return seed


def _claim_final_once(seed: int) -> Path:
    """Atomically consume the director-held one-shot final-evaluation lock.

    The lock path must be supplied outside the campaign directory through
    SEVEN_CELL_FINAL_LOCK_PATH. Creation uses O_EXCL, so a second call fails even
    across fresh Python processes. The seed itself is never written to disk.
    """
    raw = os.environ.get("SEVEN_CELL_FINAL_LOCK_PATH")
    if not raw:
        raise PermissionError(
            "Final evaluation requires director-held SEVEN_CELL_FINAL_LOCK_PATH."
        )
    lock_path = Path(raw).expanduser().resolve()
    package_dir = Path(__file__).resolve().parent
    if lock_path == package_dir or package_dir in lock_path.parents:
        raise PermissionError(
            "Final lock must live outside the campaign package directory."
        )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags, 0o600)
    except FileExistsError as exc:
        raise PermissionError(
            f"Final evaluation already consumed: lock exists at {lock_path}"
        ) from exc

    seed_fp = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:16]
    payload = (
        "Direct-7Cell final evaluation consumed\n"
        f"unix_time={time.time():.6f}\n"
        f"final_seed_sha256_prefix={seed_fp}\n"
    ).encode("utf-8")
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    return lock_path


# =============================================================================
# Campaign-source access guard
# =============================================================================
_ALLOWED_TRAIN_PREPARE7_IMPORTS = frozenset(
    {
        "B",
        "BANDWIDTH_HZ",
        "K_TOTAL",
        "K_USERS",
        "M_TX",
        "N_RX",
        "NOISE_POWER_W",
        "P_MAX_W",
        "SEED_TRAIN",
        "TORCH_SEED",
        "TrainingChannelStream",
    }
)
_FORBIDDEN_TRAIN_IDENTIFIERS = frozenset(
    {
        "_SELECT_H",
        "_BASELINE_SELECT_MEAN",
        "sample_channels",
        "evaluate_selection",
        "score_precomputed_grid",
        "evaluate_final",
        "run_mqft_reference",
        "mqft_solve_one",
        "_rng",
    }
)
_FORBIDDEN_TRAIN_SOURCE_FRAGMENTS = (
    "__globals__",
    "sys.modules",
    "importlib",
    "__import__",
)


def audit_train_source(path: Optional[str | Path] = None) -> None:
    """Enforce the Baby-style prepare7 import boundary on mutable train.py.

    This is a protocol guard, not a hostile-code sandbox. Its purpose is to catch
    direct/dynamic access paths that could mutate evaluator globals in-process.
    """
    train_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else Path(__file__).resolve().with_name("train.py")
    )
    if not train_path.is_file():
        raise RuntimeError(f"FAIL: mutable train.py not found at {train_path}")

    source = train_path.read_text(encoding="utf-8")
    for frag in _FORBIDDEN_TRAIN_SOURCE_FRAGMENTS:
        if frag in source:
            raise RuntimeError(
                f"FAIL: train.py uses forbidden evaluator-circumvention token {frag!r}"
            )

    try:
        tree = ast.parse(source, filename=str(train_path))
    except SyntaxError as exc:
        raise RuntimeError(f"FAIL: train.py syntax error: {exc}") from exc

    imported_from_prepare = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "prepare7" or alias.name.startswith("prepare7."):
                    raise RuntimeError(
                        "FAIL: train.py may not import the prepare7 module object; "
                        "use only the frozen from-prepare7 whitelist."
                    )
        elif isinstance(node, ast.ImportFrom) and node.module == "prepare7":
            for alias in node.names:
                if alias.name == "*" or alias.name not in _ALLOWED_TRAIN_PREPARE7_IMPORTS:
                    raise RuntimeError(
                        "FAIL: forbidden import from prepare7: "
                        f"{alias.name!r}; allowed={sorted(_ALLOWED_TRAIN_PREPARE7_IMPORTS)}"
                    )
                imported_from_prepare.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_TRAIN_IDENTIFIERS:
            raise RuntimeError(
                f"FAIL: train.py references forbidden evaluator identifier {node.id!r}"
            )
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_TRAIN_IDENTIFIERS:
            raise RuntimeError(
                f"FAIL: train.py references forbidden evaluator attribute {node.attr!r}"
            )
        elif (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value == SEED_SELECT
        ):
            raise RuntimeError(
                f"FAIL: train.py contains forbidden literal selection seed {SEED_SELECT}"
            )

    # Imported evaluator names may be called/read, but never rebound or shadowed.
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            else:
                targets = [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in imported_from_prepare:
                raise RuntimeError(
                    f"FAIL: train.py shadows imported evaluator name {node.name!r}"
                )
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and sub.id in imported_from_prepare:
                    raise RuntimeError(
                        f"FAIL: train.py rebinds imported evaluator name {sub.id!r}"
                    )


# =============================================================================
# Geometry + channel generation
# =============================================================================
def _drop_users_hex(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform drops over the full regular serving hexagon.

    Uses six equal-area triangles with reflected barycentric coordinates.
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
    """Generate channels from one sequential RNG stream.

    Frozen draw order:
      1) all user positions,
      2) all real Rayleigh Gaussian draws,
      3) all imaginary Rayleigh Gaussian draws.
    """
    if batch <= 0:
        raise ValueError("batch must be positive")

    offsets = _drop_users_hex(batch * B * K_USERS, rng).reshape(batch, B, K_USERS)
    user_pos = BS_POS[None, :, None] + offsets  # [T,B,K]

    # Each serving cell uses the same center-selected periodic image of each logical BS.
    dist = np.abs(
        user_pos[:, :, :, None] - CELL_IMAGE[None, :, None, :]
    )  # [T,B,K,Bp]
    dist = np.transpose(dist, (0, 2, 1, 3))  # [T,K,B,Bp]

    # Part-II propagation law: amplitude coefficient (1 + d/d0)^(-zeta/2).
    path_amp = (1.0 + dist / D0_M) ** (-PATHLOSS_EXPONENT / 2.0)

    shape = (batch, K_USERS, B, B, N_RX, M_TX)
    real = rng.normal(0.0, 1.0, size=shape)
    imag = rng.normal(0.0, 1.0, size=shape)
    fading = (real + 1j * imag) / math.sqrt(2.0)

    H = path_amp[..., None, None] * fading
    np_dtype = np.complex128 if dtype == torch.complex128 else np.complex64
    return torch.from_numpy(H.astype(np_dtype, copy=False))


def sample_channels(
    batch: int,
    seed: int,
    *,
    dtype: torch.dtype = EVAL_COMPLEX_DTYPE,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return _sample_channels_from_rng(batch, rng, dtype=dtype)


class TrainingChannelStream:
    """Fresh training channels from the one frozen training RNG stream."""

    def __init__(self, seed: int = SEED_TRAIN) -> None:
        if int(seed) != SEED_TRAIN:
            raise ValueError(
                f"TrainingChannelStream seed is frozen to SEED_TRAIN={SEED_TRAIN}; "
                f"got {seed}"
            )
        self._rng = np.random.default_rng(SEED_TRAIN)

    def sample(
        self,
        batch: int,
        *,
        dtype: torch.dtype = torch.complex64,
    ) -> torch.Tensor:
        return _sample_channels_from_rng(batch, self._rng, dtype=dtype)


# Selection set is materialized once and reused for every Kq.
_SELECT_H = sample_channels(N_SELECT, SEED_SELECT, dtype=EVAL_COMPLEX_DTYPE)


# =============================================================================
# Exact evaluator physics
# =============================================================================
def _check_H(H: torch.Tensor) -> torch.Tensor:
    expected_tail = (K_USERS, B, B, N_RX, M_TX)
    if H.ndim != 6 or tuple(H.shape[1:]) != expected_tail:
        raise ValueError(
            f"H shape {tuple(H.shape)} != [batch,{','.join(map(str, expected_tail))}]"
        )
    if not torch.is_complex(H):
        raise ValueError("H must be a complex tensor")
    return H.to(device="cpu", dtype=EVAL_COMPLEX_DTYPE)


def _received_stream_vectors(H: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """S[t,k,b,j,c,n] = H[t,k,b,c,:,:] @ V[t,j,c,:]."""
    return torch.einsum("tkbcnm,tjcm->tkbjcn", H, V)


def mmse_rates_mbps(H: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Authoritative MMSE-combined user rates in Mbps."""
    H = _check_H(H)
    V = V.to(device="cpu", dtype=EVAL_COMPLEX_DTYPE)
    batch = H.shape[0]

    S = _received_stream_vectors(H, V)  # [T,K,B,K,B,N]

    desired = torch.empty(
        (batch, K_USERS, B, N_RX), dtype=EVAL_COMPLEX_DTYPE
    )
    for k in range(K_USERS):
        for b in range(B):
            desired[:, k, b, :] = S[:, k, b, k, b, :]

    eye = torch.eye(N_RX, dtype=EVAL_COMPLEX_DTYPE).view(
        1, 1, 1, N_RX, N_RX
    )
    Ryy = torch.einsum("tkbjcn,tkbjcm->tkbnm", S, S.conj())
    Ryy = Ryy + NOISE_POWER_W * eye

    # Analytical optimal linear/MMSE combiner; no artificial jitter.
    u = torch.linalg.solve(Ryy, desired.unsqueeze(-1)).squeeze(-1)

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
    denominator = interference + noise_out

    # A valid zero transmit beam gives desired=0 -> u=0. In that exact case the
    # post-combiner expression is 0/0 although Part-II Eq. (1) is exactly SINR=0.
    zero_signal = signal == 0
    if bool(((denominator <= 0) & (~zero_signal)).any()):
        raise RuntimeError("FAIL: non-positive MMSE SINR denominator")
    safe_denominator = torch.where(
        zero_signal, torch.ones_like(denominator), denominator
    )
    sinr = signal / safe_denominator
    rates = (torch.log2(1.0 + sinr) * (BANDWIDTH_HZ / 1e6)).real
    if not bool(torch.isfinite(rates).all()):
        raise RuntimeError("FAIL: evaluator produced non-finite user rate")
    return rates, u


def slqp_from_rates(
    rates_mbps: torch.Tensor,
    Kq: int | torch.Tensor,
) -> torch.Tensor:
    """Exact network-wide sum of the Kq smallest rates."""
    if rates_mbps.ndim == 3:
        flat = rates_mbps.reshape(rates_mbps.shape[0], K_TOTAL)
    elif rates_mbps.ndim == 2 and rates_mbps.shape[1] == K_TOTAL:
        flat = rates_mbps
    else:
        raise ValueError(
            f"rates shape {tuple(rates_mbps.shape)} is incompatible with K_TOTAL={K_TOTAL}"
        )

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


def evaluate_slqp(
    H: torch.Tensor,
    V: torch.Tensor,
    Kq: int | torch.Tensor,
) -> torch.Tensor:
    rates, _ = mmse_rates_mbps(H, V)
    return slqp_from_rates(rates, Kq)


# =============================================================================
# Equal-power dominant-direct-channel-SVD normalization baseline
# =============================================================================
def equal_power_svd_beamformers(H: torch.Tensor) -> torch.Tensor:
    """Pmax/5 per stream along the dominant direct-channel right singular vector."""
    H = _check_H(H)
    direct = torch.stack(
        [H[:, :, b, b, :, :] for b in range(B)], dim=2
    )  # [T,K,B,N,M]
    _, _, Vh = torch.linalg.svd(direct, full_matrices=False)
    vmax = Vh[..., 0, :].conj()
    return SQRT_BASELINE_STREAM_POWER_W * vmax


def _baseline_means(H: torch.Tensor) -> Dict[int, float]:
    with torch.no_grad():
        Vb = equal_power_svd_beamformers(H)
        rates, _ = mmse_rates_mbps(H, Vb)
        out = {
            kq: float(slqp_from_rates(rates, kq).mean().item())
            for kq in KQ_VALUES
        }
    for kq, val in out.items():
        if not math.isfinite(val) or val <= 0.0:
            raise RuntimeError(
                f"FAIL: non-finite/non-positive baseline mean at Kq={kq}: {val}"
            )
    return out


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
    user_power = V128.abs().square().sum(dim=-1)  # [T,K,B]
    bs_power = user_power.sum(dim=1)  # [T,B]
    max_bs_power = float(bs_power.max().item())
    max_user_power = float(user_power.max().item())
    allowed = P_MAX_W * (1.0 + POWER_REL_TOL)
    if bool((bs_power > allowed).any()):
        raise RuntimeError(
            "FAIL: per-BS total-power violation: "
            f"max sum_k ||v_k,b||^2={max_bs_power:.12g} W "
            f"> Pmax*(1+tol)={allowed:.12g} W"
        )
    return {
        "max_bs_power_w": max_bs_power,
        "max_user_power_w": max_user_power,
    }


def _snapshot_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _assert_state_unchanged(
    model: torch.nn.Module,
    before: Dict[str, torch.Tensor],
) -> None:
    after = model.state_dict()
    if set(after) != set(before):
        raise RuntimeError(
            "FAIL: inference contract violated: model state_dict keys changed"
        )
    for k, v0 in before.items():
        v1 = after[k].detach().cpu()
        if not torch.equal(v0, v1):
            raise RuntimeError(
                f"FAIL: inference contract violated: model state changed at '{k}'"
            )


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

    # Unmeasured warm-up.
    with torch.no_grad():
        wbatch = min(8, H.shape[0])
        _ = model(H[:wbatch].clone(), _model_kq(wbatch, 1))
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
        validity["max_bs_power_w"] = max(
            validity["max_bs_power_w"], vd["max_bs_power_w"]
        )
        validity["max_user_power_w"] = max(
            validity["max_user_power_w"], vd["max_user_power_w"]
        )

        with torch.no_grad():
            val = float(evaluate_slqp(H, V, kq).mean().item())
        base = float(baseline_means[kq])
        if not math.isfinite(val):
            raise RuntimeError(f"FAIL: non-finite mean SLqP at Kq={kq}: {val}")
        if not math.isfinite(base) or base <= 0.0:
            raise RuntimeError(
                f"FAIL: invalid baseline denominator at Kq={kq}: {base}"
            )
        score = val / base
        if not math.isfinite(score):
            raise RuntimeError(f"FAIL: non-finite normalized score at Kq={kq}")
        raw_slqp_mbps_by_kq[kq] = val
        score_by_kq[kq] = score

    _assert_state_unchanged(model, state_before)

    if enforce_budget and INF_BUDGET_S is not None and inference_time_s > INF_BUDGET_S:
        raise RuntimeError(
            f"FAIL: inference budget violated: {inference_time_s:.6f}s > "
            f"{INF_BUDGET_S:.6f}s for all {H.shape[0] * len(KQ_VALUES)} tasks"
        )

    heldoutscore = float(np.mean([score_by_kq[kq] for kq in KQ_VALUES]))
    if not math.isfinite(heldoutscore):
        raise RuntimeError("FAIL: HELDOUTSCORE is NaN or Inf")
    result: Dict[str, object] = {
        "HELDOUTSCORE": heldoutscore,
        "score_by_kq": score_by_kq,
        "raw_slqp_mbps_by_kq": raw_slqp_mbps_by_kq,
        "baseline_slqp_mbps_by_kq": dict(baseline_means),
        "inference_time_s": inference_time_s,
        "inference_budget_s": INF_BUDGET_S,
        "inference_budget_pass": (
            INF_BUDGET_S is None or inference_time_s <= INF_BUDGET_S
        ),
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

    return result



def score_precomputed_grid(
    H: torch.Tensor,
    V_grid: torch.Tensor,
    baseline_means: Dict[int, float],
    *,
    inference_time_s: float,
    enforce_budget: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Authoritatively score precomputed V(H,Kq) without importing train.py.

    V_grid must have shape [35, batch, 5, 7, 8], ordered exactly by KQ_VALUES.
    This is the scoring half of the process-separated director protocol.
    """
    H = _check_H(H)
    expected = (len(KQ_VALUES), H.shape[0], K_USERS, B, M_TX)
    if tuple(V_grid.shape) != expected:
        raise RuntimeError(
            f"FAIL: precomputed V grid shape {tuple(V_grid.shape)} != {expected}"
        )
    if not torch.is_complex(V_grid):
        raise RuntimeError("FAIL: precomputed V grid must be complex")
    if not bool(torch.isfinite(V_grid.real).all() and torch.isfinite(V_grid.imag).all()):
        raise RuntimeError("FAIL: precomputed V grid contains NaN or Inf")
    if not math.isfinite(float(inference_time_s)) or float(inference_time_s) < 0.0:
        raise RuntimeError("FAIL: invalid inference_time_s")

    score_by_kq: Dict[int, float] = {}
    raw_slqp_mbps_by_kq: Dict[int, float] = {}
    validity = {"max_bs_power_w": 0.0, "max_user_power_w": 0.0}

    for idx, kq in enumerate(KQ_VALUES):
        V = V_grid[idx].to(device="cpu", dtype=EVAL_COMPLEX_DTYPE)
        vd = _validate_model_output(V, H.shape[0])
        validity["max_bs_power_w"] = max(
            validity["max_bs_power_w"], vd["max_bs_power_w"]
        )
        validity["max_user_power_w"] = max(
            validity["max_user_power_w"], vd["max_user_power_w"]
        )

        with torch.no_grad():
            val = float(evaluate_slqp(H, V, kq).mean().item())
        base = float(baseline_means[kq])
        if not math.isfinite(val):
            raise RuntimeError(f"FAIL: non-finite mean SLqP at Kq={kq}: {val}")
        if not math.isfinite(base) or base <= 0.0:
            raise RuntimeError(
                f"FAIL: invalid baseline denominator at Kq={kq}: {base}"
            )
        score = val / base
        if not math.isfinite(score):
            raise RuntimeError(f"FAIL: non-finite normalized score at Kq={kq}")
        raw_slqp_mbps_by_kq[kq] = val
        score_by_kq[kq] = score

    if enforce_budget and INF_BUDGET_S is not None and inference_time_s > INF_BUDGET_S:
        raise RuntimeError(
            f"FAIL: inference budget violated: {inference_time_s:.6f}s > "
            f"{INF_BUDGET_S:.6f}s for {H.shape[0] * len(KQ_VALUES)} tasks"
        )

    heldoutscore = float(np.mean([score_by_kq[kq] for kq in KQ_VALUES]))
    if not math.isfinite(heldoutscore):
        raise RuntimeError("FAIL: HELDOUTSCORE is NaN or Inf")

    result: Dict[str, object] = {
        "HELDOUTSCORE": heldoutscore,
        "score_by_kq": score_by_kq,
        "raw_slqp_mbps_by_kq": raw_slqp_mbps_by_kq,
        "baseline_slqp_mbps_by_kq": dict(baseline_means),
        "inference_time_s": float(inference_time_s),
        "inference_budget_s": INF_BUDGET_S,
        "inference_budget_pass": (
            INF_BUDGET_S is None or inference_time_s <= INF_BUDGET_S
        ),
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

    return result

def evaluate_selection(
    model: torch.nn.Module,
    *,
    verbose: bool = True,
) -> Dict[str, object]:
    audit_train_source()
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
    """Measure only the complete (H,Kq)->V grid, excluding scoring and I/O."""
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
    """Exactly-once protected final evaluation after champion freeze."""
    if not authorization:
        raise PermissionError(
            "Final evaluation is protocol-protected. Set authorization=True only "
            "after the campaign champion is frozen."
        )
    audit_train_source()
    seed = _final_seed()
    lock_path = _claim_final_once(seed)
    print(f"FINAL_LOCK {lock_path}")
    H_final = sample_channels(N_FINAL, seed, dtype=EVAL_COMPLEX_DTYPE)
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
    """Closed-form MQFT auxiliary-variable update for one channel realization."""
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


def _mqft_v_update_cvxpy(
    H_one: torch.Tensor,
    chis: np.ndarray,
    Kq: int,
) -> torch.Tensor:
    """Fixed-X concave MQFT beamformer update under per-BS total-power limits."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Offline MQFT requires cvxpy with CLARABEL or SCS; it is not needed "
            "for training or learned-model inference."
        ) from exc

    Hn = H_one.detach().cpu().numpy().astype(np.complex128, copy=False)
    Vvar = cp.Variable((K_TOTAL, M_TX), complex=True)
    rates = []
    constraints = []

    def sidx(k: int, b: int) -> int:
        return k * B + b

    for b in range(B):
        rows = [sidx(k, b) for k in range(K_USERS)]
        constraints.append(cp.sum_squares(cp.abs(Vvar[rows, :])) <= P_MAX_W)

    for k in range(K_USERS):
        for b in range(B):
            chi = chis[k, b]
            des = sidx(k, b)
            hdes = Hn[k, b, b]
            coeff_des = chi.conj() @ hdes

            q = 1.0 - NOISE_POWER_W * float(np.vdot(chi, chi).real)
            q = q + 2.0 * cp.real(coeff_des @ Vvar[des, :])

            for j in range(K_USERS):
                for c in range(B):
                    if j == k and c == b:
                        continue
                    coeff = chi.conj() @ Hn[k, b, c]
                    q = q - cp.square(cp.abs(coeff @ Vvar[sidx(j, c), :]))

            constraints.append(q >= 1e-12)
            rates.append(cp.log(q))  # natural-log scale; argmax unchanged

    rvec = cp.hstack(rates)
    obj = cp.sum(rvec) if Kq == K_TOTAL else cp.sum_smallest(rvec, Kq)
    prob = cp.Problem(cp.Maximize(obj), constraints)

    solved = False
    Vf = None
    last_max_bs_power = None
    for solver in ("CLARABEL", "SCS"):
        try:
            if solver == "SCS":
                prob.solve(
                    solver=solver,
                    verbose=False,
                    eps_abs=1e-9,
                    eps_rel=1e-9,
                    max_iters=1_000_000,
                    warm_start=False,
                )
            else:
                prob.solve(solver=solver, verbose=False)
            if (
                prob.status in ("optimal", "optimal_inaccurate")
                and Vvar.value is not None
            ):
                Vf_candidate = np.asarray(
                    Vvar.value, dtype=np.complex128
                ).reshape(K_USERS, B, M_TX)
                bs_power = np.sum(np.abs(Vf_candidate) ** 2, axis=(0, 2))
                last_max_bs_power = float(np.max(bs_power))
                if last_max_bs_power <= P_MAX_W * (1.0 + 1e-6):
                    Vf = Vf_candidate
                    solved = True
                    break
        except Exception:
            continue
    if not solved:
        detail = (
            f"; max BS total power={last_max_bs_power} W"
            if last_max_bs_power is not None
            else ""
        )
        raise RuntimeError(
            f"MQFT fixed-X V update failed; CVXPY status={prob.status}{detail}"
        )

    return torch.from_numpy(Vf)


def mqft_solve_one(
    H_one: torch.Tensor,
    Kq: int,
    *,
    rel_tol: float = 1e-6,
    max_iter: int = 25,
) -> MQFTResult:
    """Frozen MQFT reference: SVD initialization, tol 1e-6, max 25 iterations."""
    if Kq not in MQFT_KQ_VALUES:
        raise ValueError(
            f"MQFT reference is frozen only for Kq in {MQFT_KQ_VALUES}; got {Kq}"
        )

    H_one = _check_H(H_one.unsqueeze(0))[0]
    V = equal_power_svd_beamformers(H_one.unsqueeze(0))[0]
    f_prev = float(
        evaluate_slqp(H_one.unsqueeze(0), V.unsqueeze(0), Kq).item()
    )
    trajectory = [f_prev]
    n_decreases = 0

    for it in range(1, max_iter + 1):
        chis = _mqft_auxiliaries(H_one, V)
        V = _mqft_v_update_cvxpy(H_one, chis, Kq)
        f_new = float(
            evaluate_slqp(H_one.unsqueeze(0), V.unsqueeze(0), Kq).item()
        )
        if f_new < f_prev * (1.0 - 1e-9):
            n_decreases += 1
        trajectory.append(f_new)

        rel = abs(f_new - f_prev) / max(abs(f_prev), 1e-12)
        if rel < rel_tol:
            return MQFTResult(
                V=V,
                trajectory_mbps=tuple(trajectory),
                converged=True,
                iterations=it,
                n_decreases=n_decreases,
            )
        f_prev = f_new

    return MQFTResult(
        V=V,
        trajectory_mbps=tuple(trajectory),
        converged=False,
        iterations=max_iter,
        n_decreases=n_decreases,
    )


def run_mqft_reference(
    *,
    dataset: str = "selection",
    authorization: bool = False,
    cache_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Offline MQFT reference on the frozen six-point Kq grid only."""
    if dataset == "selection":
        Hset = _SELECT_H
    elif dataset == "final":
        if not authorization:
            raise PermissionError(
                "Final MQFT reference is protected until champion freeze"
            )
        Hset = sample_channels(N_FINAL, _final_seed(), dtype=EVAL_COMPLEX_DTYPE)
    else:
        raise ValueError("dataset must be 'selection' or 'final'")

    nk = len(MQFT_KQ_VALUES)
    values = np.empty((Hset.shape[0], nk), dtype=np.float64)
    converged = np.zeros((Hset.shape[0], nk), dtype=np.bool_)
    iterations = np.zeros((Hset.shape[0], nk), dtype=np.int64)
    decreases = np.zeros((Hset.shape[0], nk), dtype=np.int64)

    for n in range(Hset.shape[0]):
        for col, kq in enumerate(MQFT_KQ_VALUES):
            res = mqft_solve_one(Hset[n], kq)
            values[n, col] = res.trajectory_mbps[-1]
            converged[n, col] = res.converged
            iterations[n, col] = res.iterations
            decreases[n, col] = res.n_decreases
            print(
                f"MQFT dataset={dataset} drop={n+1}/{Hset.shape[0]} Kq={kq} "
                f"SLqP={values[n,col]:.6f} converged={bool(converged[n,col])} "
                f"iters={iterations[n,col]} decreases={decreases[n,col]}"
            )

    out = {
        "kq_values": MQFT_KQ_VALUES,
        "mean_slqp_mbps_by_kq": {
            kq: float(values[:, i].mean())
            for i, kq in enumerate(MQFT_KQ_VALUES)
        },
        "converged_fraction_by_kq": {
            kq: float(converged[:, i].mean())
            for i, kq in enumerate(MQFT_KQ_VALUES)
        },
        "cap_hits_by_kq": {
            kq: int((~converged[:, i]).sum())
            for i, kq in enumerate(MQFT_KQ_VALUES)
        },
        "decreases_by_kq": {
            kq: int(decreases[:, i].sum())
            for i, kq in enumerate(MQFT_KQ_VALUES)
        },
        "values": values,
        "converged": converged,
        "iterations": iterations,
        "decreases": decreases,
    }

    if cache_path is not None:
        cache_path = Path(cache_path)
        np.savez_compressed(
            cache_path,
            kq_values=np.asarray(MQFT_KQ_VALUES, dtype=np.int64),
            values=values,
            converged=converged,
            iterations=iterations,
            decreases=decreases,
            dataset=dataset,
        )
    return out


# =============================================================================
# Lightweight integrity audit
# =============================================================================
def self_audit() -> None:
    assert B == 7 and K_USERS == 5 and K_TOTAL == 35
    assert M_TX == 8 and N_RX == 2
    assert np.allclose(np.abs(SHIFTS[1:]), 2.0 * APOTHEM_M * np.sqrt(7.0))
    assert CELL_IMAGE.shape == (B, B)
    assert abs(P_MAX_W - 10.0 ** 1.3) < 1e-12
    assert _SELECT_H.shape == (
        N_SELECT,
        K_USERS,
        B,
        B,
        N_RX,
        M_TX,
    )
    assert _SELECT_H.dtype == EVAL_COMPLEX_DTYPE

    with torch.no_grad():
        Vb = equal_power_svd_beamformers(_SELECT_H[:2])
        _validate_model_output(Vb, 2)
        rates, u = mmse_rates_mbps(_SELECT_H[:2], Vb)
        assert rates.shape == (2, K_USERS, B)
        assert u.shape == (2, K_USERS, B, N_RX)
        assert bool(torch.isfinite(rates).all())

        # Regression test: an exactly switched-off feasible stream must score 0,
        # not NaN via a post-combiner 0/0.
        Vz = Vb.clone()
        Vz[:, 0, 0, :] = 0.0
        rz, _ = mmse_rates_mbps(_SELECT_H[:2], Vz)
        assert bool(torch.isfinite(rz).all())
        assert bool((rz[:, 0, 0] == 0.0).all())

        for kq in KQ_VALUES:
            f = slqp_from_rates(rates, kq)
            assert f.shape == (2,)
            assert bool(torch.isfinite(f).all())

    print("prepare7.py self_audit: PASS")


if __name__ == "__main__":
    self_audit()
