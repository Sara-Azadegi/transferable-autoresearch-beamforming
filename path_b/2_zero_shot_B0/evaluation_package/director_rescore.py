"""Immutable director-side scorer for Direct 7-cell Autoresearch.

Authoritative evaluation is deliberately split across processes:

  parent/director process
      verifies the external hashes + package Python-file allow-list;
  inference worker (Process A)
      imports mutable train.py, loads the checkpoint, computes V(H,Kq), and writes
      ONLY precomputed beamformers plus trusted timing metadata;
  scoring worker (Process B)
      NEVER imports train.py, regenerates the authoritative channels independently,
      validates the saved beamformers, and computes HELDOUTSCORE with prepare7.py.

This separation prevents mutable train.py from monkey-patching the evaluator's
math/timer in the process that produces the authoritative score.

Usage (selection):
    python director_rescore.py last_model.pt \
        --hash-file /path/outside/campaign/Direct-7Cell-DIRECTOR-HASHES.sha256

Usage (protected final, after champion freeze):
    python director_rescore.py champion.pt --dataset final --authorization \
        --hash-file /path/outside/campaign/Direct-7Cell-DIRECTOR-HASHES.sha256
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# EARLY package-file allow-list.
# Do this before importing torch or any other third-party module so a stray
# package-local torch.py/helper module cannot be imported first.
# -----------------------------------------------------------------------------
import os as _os
import sys as _sys
import importlib.machinery as _machinery

_PACKAGE_DIR_RAW = _os.path.dirname(_os.path.abspath(__file__))
_ALLOWED_PY_REL = frozenset({"prepare7.py", "train.py", "director_rescore.py"})
_IMPORTABLE_SUFFIXES = tuple(_machinery.all_suffixes())
_EXTRA_FORBIDDEN_PACKAGE_SUFFIXES = (".pyw", ".pth", ".zip")


def _is_link_or_junction(path: str) -> bool:
    isjunction = getattr(_os.path, "isjunction", None)
    return _os.path.islink(path) or bool(isjunction and isjunction(path))


def _early_python_file_allowlist() -> None:
    found: set[str] = set()
    for root, dirs, files in _os.walk(_PACKAGE_DIR_RAW):
        # Normal interpreter caches are harmless here because every authoritative
        # process runs with an external fresh pycache_prefix. Do not descend into them.
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for d in dirs:
            full_dir = _os.path.join(root, d)
            rel_dir = _os.path.relpath(full_dir, _PACKAGE_DIR_RAW).replace(_os.sep, "/")
            if _is_link_or_junction(full_dir):
                raise RuntimeError(
                    f"PACKAGE ALLOW-LIST FAIL: linked/junction directory {rel_dir!r}"
                )
        for name in files:
            full = _os.path.join(root, name)
            rel = _os.path.relpath(full, _PACKAGE_DIR_RAW).replace(_os.sep, "/")
            if _is_link_or_junction(full) or not _os.path.isfile(full):
                raise RuntimeError(
                    f"PACKAGE ALLOW-LIST FAIL: {rel!r} must be a regular non-linked file"
                )
            lower = name.lower()
            if lower.endswith(_EXTRA_FORBIDDEN_PACKAGE_SUFFIXES):
                raise RuntimeError(
                    f"PACKAGE ALLOW-LIST FAIL: executable/importable artifact {rel!r}"
                )
            import_suffix = next(
                (suffix for suffix in _IMPORTABLE_SUFFIXES if lower.endswith(suffix.lower())),
                None,
            )
            if import_suffix is not None and not lower.endswith(".py"):
                raise RuntimeError(
                    "PACKAGE ALLOW-LIST FAIL: executable/importable artifact "
                    f"{rel!r} with import suffix {import_suffix!r}"
                )
            if not lower.endswith(".py"):
                continue
            found.add(rel)
            if rel not in _ALLOWED_PY_REL:
                raise RuntimeError(
                    "PACKAGE ALLOW-LIST FAIL: unexpected Python source file "
                    f"{rel!r}; allowed={sorted(_ALLOWED_PY_REL)}"
                )
    missing = _ALLOWED_PY_REL.difference(found)
    if missing:
        raise RuntimeError(
            f"PACKAGE ALLOW-LIST FAIL: required Python files missing: {sorted(missing)}"
        )


_early_python_file_allowlist()
_sys.dont_write_bytecode = True

# Safe to import ordinary/third-party modules only after the allow-list passes.
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Dict

PACKAGE_DIR = Path(_PACKAGE_DIR_RAW)
LOAD_CANDIDATE_BUDGET_S = 30.0
INFER_WORKER_WALL_TIMEOUT_S = 90.0
SCORE_WORKER_WALL_TIMEOUT_S = 300.0
_ARTIFACT_MAGIC = "DIRECT7_PRECOMPUTED_V_V1"


def _read_hashes(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed director hash line: {raw!r}")
        digest, name = parts
        expected[name.strip()] = digest.strip().lower()
    return expected


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_REQUIRED_IMMUTABLE_HASHES = frozenset(
    {"prepare7.py", "program.md", "director_rescore.py", "requirements.txt"}
)


def _validate_expected_hashes(expected: dict[str, str]) -> dict[str, str]:
    missing = _REQUIRED_IMMUTABLE_HASHES.difference(expected)
    if missing:
        raise RuntimeError(f"Director hash record missing entries: {sorted(missing)}")
    clean: dict[str, str] = {}
    for name in sorted(_REQUIRED_IMMUTABLE_HASHES):
        digest = str(expected[name]).strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Malformed SHA-256 digest for {name!r}")
        clean[name] = digest
    return clean


def load_director_hashes(hash_file: Path) -> dict[str, str]:
    """Read the external trust anchor exactly once in the authoritative parent."""
    hash_file = hash_file.expanduser().resolve()
    if PACKAGE_DIR == hash_file.parent or PACKAGE_DIR in hash_file.parents:
        raise RuntimeError(
            "Director hash file must be held outside the campaign package directory."
        )
    if not hash_file.is_file():
        raise RuntimeError(f"Director hash file not found: {hash_file}")
    return _validate_expected_hashes(_read_hashes(hash_file))


def verify_expected_hashes(expected: dict[str, str]) -> None:
    """Verify immutable package files against already-trusted in-memory digests."""
    _early_python_file_allowlist()
    expected = _validate_expected_hashes(expected)
    for name in sorted(_REQUIRED_IMMUTABLE_HASHES):
        path = PACKAGE_DIR / name
        got = _sha256(path)
        want = expected[name]
        if got != want:
            raise RuntimeError(
                f"IMMUTABLE HASH FAIL: {name}: got {got}, expected {want}"
            )
    print("IMMUTABLE_HASHES PASS")


def verify_director_hashes(hash_file: Path) -> dict[str, str]:
    """Load the external hashes once, verify them, and return the trusted copy."""
    expected = load_director_hashes(hash_file)
    verify_expected_hashes(expected)
    return expected


def _authoritative_H(prepare7, dataset: str):
    if dataset == "selection":
        return prepare7._SELECT_H.detach().clone()
    if dataset == "final":
        seed = prepare7._final_seed()
        return prepare7.sample_channels(
            prepare7.N_FINAL, seed, dtype=prepare7.EVAL_COMPLEX_DTYPE
        )
    raise ValueError("dataset must be 'selection' or 'final'")


def _snapshot_state(model) -> Dict[str, object]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _assert_state_unchanged(model, before: Dict[str, object], torch) -> None:
    after = model.state_dict()
    if set(after) != set(before):
        raise RuntimeError("FAIL: inference contract violated: model state_dict keys changed")
    for k, v0 in before.items():
        v1 = after[k].detach().cpu()
        if not torch.equal(v0, v1):
            raise RuntimeError(
                f"FAIL: inference contract violated: model state changed at {k!r}"
            )


def _worker_infer(args) -> None:
    """Process A: mutable train.py is imported here; no score is computed here."""
    expected_hashes = json.loads(args.expected_hashes_json)
    verify_expected_hashes(expected_hashes)

    if str(PACKAGE_DIR) not in _sys.path:
        _sys.path.insert(0, str(PACKAGE_DIR))

    # Import evaluator and bind trusted inputs/timer BEFORE importing mutable train.py.
    prepare7 = importlib.import_module("prepare7")
    prepare7.audit_train_source(PACKAGE_DIR / "train.py")
    print("TRAIN_ACCESS_AUDIT PASS")

    import numpy as np
    import torch

    # Bind trusted library callables/types BEFORE mutable train.py is imported.
    trusted_set_threads = torch.set_num_threads
    trusted_get_threads = torch.get_num_threads
    trusted_perf_counter = time.perf_counter
    trusted_full = torch.full
    trusted_no_grad = torch.no_grad
    trusted_clone = torch.Tensor.clone
    trusted_is_complex = torch.is_complex
    trusted_complex128 = torch.complex128
    tensor_type = torch.Tensor
    trusted_np_stack = np.stack
    trusted_np_asarray = np.asarray
    trusted_np_savez = np.savez
    trusted_np_float64 = np.float64
    trusted_np_int64 = np.int64

    trusted_set_threads(20)
    if trusted_get_threads() != 20:
        raise RuntimeError("FAIL: could not set required 20 PyTorch intra-op threads")

    H = _authoritative_H(prepare7, args.dataset)

    # Mutable code enters only after all trusted evaluator inputs/bindings above exist.
    train = importlib.import_module("train")
    # Restore the frozen thread count in case import-time mutable code changed it.
    trusted_set_threads(20)
    if trusted_get_threads() != 20:
        raise RuntimeError("FAIL: train.py changed the required thread configuration")
    if not hasattr(train, "load_candidate"):
        raise RuntimeError(
            "FAIL: train.py must expose load_candidate(checkpoint_path) for "
            "director inference."
        )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"FAIL: checkpoint not found: {checkpoint}")

    t_load = trusted_perf_counter()
    model = train.load_candidate(str(checkpoint))
    load_s = trusted_perf_counter() - t_load
    print(f"LOAD_CANDIDATE_S {load_s:.6f}")
    if load_s > LOAD_CANDIDATE_BUDGET_S:
        raise RuntimeError(
            f"FAIL: load_candidate time {load_s:.6f}s > "
            f"{LOAD_CANDIDATE_BUDGET_S:.6f}s"
        )

    model.eval()
    state_before = _snapshot_state(model)
    batch = int(H.shape[0])

    # Untimed warm-up, exactly as in the original evaluator contract.
    wbatch = min(8, batch)
    with trusted_no_grad():
        _ = model(
            trusted_clone(H[:wbatch]),
            trusted_full((wbatch,), 1, dtype=torch.long),
        )
    _assert_state_unchanged(model, state_before, torch)

    outputs = []
    inference_time_s = 0.0
    for kq in prepare7.KQ_VALUES:
        H_in = trusted_clone(H)
        kq_input = trusted_full((batch,), int(kq), dtype=torch.long)
        t0 = trusted_perf_counter()
        with trusted_no_grad():
            V = model(H_in, kq_input)
        if type(V) is not tensor_type:
            raise RuntimeError("FAIL: model output must be an exact torch.Tensor")
        expected = (batch, prepare7.K_USERS, prepare7.B, prepare7.M_TX)
        if tuple(V.shape) != expected:
            raise RuntimeError(f"FAIL: model output shape {tuple(V.shape)} != {expected}")
        if not trusted_is_complex(V):
            raise RuntimeError("FAIL: model output V must be complex")
        V_np = V.detach().to(device="cpu", dtype=trusted_complex128).numpy().copy()
        inference_time_s += trusted_perf_counter() - t0
        outputs.append(V_np)

    _assert_state_unchanged(model, state_before, torch)
    print(f"INFERENCE_S {inference_time_s:.6f}")
    if prepare7.INF_BUDGET_S is not None and inference_time_s > prepare7.INF_BUDGET_S:
        raise RuntimeError(
            f"FAIL: inference budget violated: {inference_time_s:.6f}s > "
            f"{prepare7.INF_BUDGET_S:.6f}s for "
            f"{batch * len(prepare7.KQ_VALUES)} tasks"
        )

    V_grid = trusted_np_stack(outputs, axis=0)
    artifact = Path(args.artifact).resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    trusted_np_savez(
        artifact,
        magic=trusted_np_asarray(_ARTIFACT_MAGIC),
        dataset=trusted_np_asarray(args.dataset),
        V_grid=V_grid,
        inference_time_s=trusted_np_asarray(inference_time_s, dtype=trusted_np_float64),
        load_candidate_s=trusted_np_asarray(load_s, dtype=trusted_np_float64),
        torch_version=trusted_np_asarray(str(torch.__version__)),
        torch_threads=trusted_np_asarray(int(trusted_get_threads()), dtype=trusted_np_int64),
    )
    print(f"PRECOMPUTED_V {artifact}")


def _worker_score(args) -> None:
    """Process B: authoritative scorer. This process NEVER imports train.py."""
    expected_hashes = json.loads(args.expected_hashes_json)
    verify_expected_hashes(expected_hashes)

    if "train" in _sys.modules:
        raise RuntimeError("FAIL: scoring process must never import train.py")
    if str(PACKAGE_DIR) not in _sys.path:
        _sys.path.insert(0, str(PACKAGE_DIR))

    prepare7 = importlib.import_module("prepare7")
    if "train" in _sys.modules:
        raise RuntimeError("FAIL: scoring process contaminated by train.py import")

    import numpy as np
    import torch

    artifact = Path(args.artifact).resolve()
    if not artifact.is_file():
        raise RuntimeError(f"FAIL: precomputed inference artifact missing: {artifact}")

    with np.load(artifact, allow_pickle=False) as data:
        magic = str(data["magic"].item())
        dataset = str(data["dataset"].item())
        if magic != _ARTIFACT_MAGIC:
            raise RuntimeError(f"FAIL: unexpected inference artifact magic {magic!r}")
        if dataset != args.dataset:
            raise RuntimeError(
                f"FAIL: artifact dataset {dataset!r} != requested {args.dataset!r}"
            )
        V_np = np.asarray(data["V_grid"])
        inference_time_s = float(data["inference_time_s"].item())
        load_s = float(data["load_candidate_s"].item())
        torch_version = str(data["torch_version"].item())
        torch_threads = int(data["torch_threads"].item())

    if not np.isfinite(inference_time_s) or inference_time_s < 0.0:
        raise RuntimeError("FAIL: invalid inference timing metadata")
    if not np.isfinite(load_s) or load_s < 0.0 or load_s > LOAD_CANDIDATE_BUDGET_S:
        raise RuntimeError("FAIL: invalid load_candidate timing metadata")
    if torch_threads != 20:
        raise RuntimeError(f"FAIL: inference worker used {torch_threads} threads, expected 20")

    H = _authoritative_H(prepare7, args.dataset)
    baseline = (
        prepare7._BASELINE_SELECT_MEAN
        if args.dataset == "selection"
        else prepare7._baseline_means(H)
    )
    V_grid = torch.from_numpy(V_np)

    result = prepare7.score_precomputed_grid(
        H,
        V_grid,
        baseline,
        inference_time_s=inference_time_s,
        enforce_budget=True,
        verbose=True,
    )
    print(f"INFER_WORKER_TORCH_VERSION {torch_version}")
    print(f"INFER_WORKER_THREADS {torch_threads}")
    print(f"LOAD_CANDIDATE_S {load_s:.6f}")
    print(f"WORKER_HELDOUTSCORE {result['HELDOUTSCORE']:.12f}")


def _run_parent(args) -> None:
    # The external trust anchor is read exactly once, before mutable Process A runs.
    # All later integrity checks use this in-memory copy, so Process A cannot make
    # tampering pass by rewriting the external .sha256 file.
    expected_hashes = verify_director_hashes(Path(args.hash_file))
    expected_hashes_json = json.dumps(expected_hashes, sort_keys=True)

    if str(PACKAGE_DIR) not in _sys.path:
        _sys.path.insert(0, str(PACKAGE_DIR))
    prepare7 = importlib.import_module("prepare7")
    prepare7.audit_train_source(PACKAGE_DIR / "train.py")
    print("TRAIN_ACCESS_AUDIT PASS")

    if args.dataset == "final":
        if not args.authorization:
            raise PermissionError(
                "Final evaluation is protocol-protected. Use --authorization only "
                "after the campaign champion is frozen."
            )
        seed = prepare7._final_seed()
        lock_path = prepare7._claim_final_once(seed)
        print(f"FINAL_LOCK {lock_path}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()

    with tempfile.TemporaryDirectory(prefix="direct7_process_isolated_") as td:
        artifact = Path(td) / "precomputed_v.npz"
        base = [
            _sys.executable,
            str(PACKAGE_DIR / "director_rescore.py"),
            str(checkpoint),
            "--expected-hashes-json",
            expected_hashes_json,
            "--dataset",
            args.dataset,
            "--artifact",
            str(artifact),
        ]

        infer_cache = Path(td) / "pycache_infer"
        score_cache = Path(td) / "pycache_score"
        infer_cmd = [
            _sys.executable, "-I", "-X", f"pycache_prefix={infer_cache}",
            *base[1:], "--worker", "infer"
        ]
        print("PROCESS_A inference worker")
        try:
            infer_run = subprocess.run(
                infer_cmd,
                cwd=str(PACKAGE_DIR),
                check=False,
                timeout=INFER_WORKER_WALL_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FAIL: inference worker exceeded safety wall timeout "
                f"{INFER_WORKER_WALL_TIMEOUT_S:.1f}s"
            ) from exc
        for line in infer_run.stdout.splitlines():
            print(f"PROCESS_A> {line}")
        for line in infer_run.stderr.splitlines():
            print(f"PROCESS_A_ERR> {line}", file=_sys.stderr)
        if infer_run.returncode != 0:
            raise RuntimeError(
                f"FAIL: inference worker exited with status {infer_run.returncode}"
            )

        # Mutable train.py has just executed in Process A. Re-verify the immutable
        # sources and package allow-list before launching the authoritative scorer.
        verify_expected_hashes(expected_hashes)
        print("POST_INFERENCE_INTEGRITY PASS")

        score_cmd = [
            _sys.executable, "-I", "-X", f"pycache_prefix={score_cache}",
            *base[1:], "--worker", "score"
        ]
        print("PROCESS_B scoring worker (train.py is never imported here)")
        try:
            score_run = subprocess.run(
                score_cmd,
                cwd=str(PACKAGE_DIR),
                check=False,
                timeout=SCORE_WORKER_WALL_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FAIL: scoring worker exceeded safety wall timeout "
                f"{SCORE_WORKER_WALL_TIMEOUT_S:.1f}s"
            ) from exc
        if score_run.stdout:
            print(score_run.stdout, end="")
        if score_run.stderr:
            print(score_run.stderr, end="", file=_sys.stderr)
        if score_run.returncode != 0:
            raise RuntimeError(
                f"FAIL: scoring worker exited with status {score_run.returncode}"
            )
        score_lines = [
            line for line in score_run.stdout.splitlines()
            if line.startswith("WORKER_HELDOUTSCORE ")
        ]
        if len(score_lines) != 1:
            raise RuntimeError("FAIL: scoring worker did not emit exactly one score")
        score = float(score_lines[0].split()[1])
        print(f"DIRECTOR_HELDOUTSCORE {score:.12f}")


def _relaunch_isolated_if_needed() -> None:
    """Make the authoritative parent run under Python isolated mode."""
    if _sys.flags.isolated:
        return
    with tempfile.TemporaryDirectory(prefix="direct7_parent_pycache_") as cache_dir:
        cmd = [
            _sys.executable,
            "-I",
            "-X",
            f"pycache_prefix={cache_dir}",
            str(PACKAGE_DIR / "director_rescore.py"),
            *_sys.argv[1:],
        ]
        completed = subprocess.run(cmd)
        raise SystemExit(completed.returncode)


def main() -> None:
    _relaunch_isolated_if_needed()
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="last_model.pt")
    parser.add_argument("--hash-file", default=None)
    parser.add_argument("--expected-hashes-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=("selection", "final"), default="selection")
    parser.add_argument("--authorization", action="store_true")
    parser.add_argument("--worker", choices=("infer", "score"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--artifact", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        if not args.artifact:
            raise RuntimeError("internal worker requires --artifact")
        if not args.expected_hashes_json:
            raise RuntimeError("internal worker requires parent-provided expected hashes")
        if args.worker == "infer":
            _worker_infer(args)
        else:
            _worker_score(args)
        return

    if not args.hash_file:
        raise RuntimeError("authoritative parent requires --hash-file")
    _run_parent(args)


if __name__ == "__main__":
    main()
