"""Recompute every reported number from the evaluator logs in verify/logs/.

Read-only: parses text logs, runs no model and no evaluator, needs only the Python standard
library (3.9+). Checks, for each evaluation, that the printed score equals the equal-weight mean
of its per-Kq normalized scores; that the three 7-cell protected final tests used identical
channels; and prints the comparisons reported in the paper.

    python verify/verify_results.py        # exit status 0 if every check passes
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LOGS = Path(__file__).resolve().parent / "logs"
MQFT_GRID = (1, 2, 4, 9, 18, 35)  # Kq grid of the 7-cell MQFT reference (prepare7.MQFT_KQ_VALUES)

# (key, label, log file, evaluation set, system)
RUNS = [
    ("2c_sel", "2-cell champion (Run-25)", "2cell_run25_selection.log", "selection", "2cell"),
    ("2c_fin", "2-cell champion (Run-25)", "2cell_run25_final_test.log", "final", "2cell"),
    ("A_sel", "Path A: direct 7-cell (Run-18)", "pathA_run18_selection.log", "selection", "7cell"),
    ("A_fin", "Path A: direct 7-cell (Run-18)", "pathA_run18_final_test.log", "final", "7cell"),
    ("B0_sel", "B0: zero-shot transfer", "B0_selection.log", "selection", "7cell"),
    ("B0_fin", "B0: zero-shot transfer", "B0_final_test.log", "final", "7cell"),
    ("B2_sel", "B2: transfer + fine-tuning", "B2_selection.log", "selection", "7cell"),
    ("B2_fin", "B2: transfer + fine-tuning", "B2_final_test.log", "final", "7cell"),
]
KQ_7 = re.compile(r"^KQ (\d+)\s+SCORE ([\d.]+)\s+RAW_SLQP_MBPS ([\d.]+)\s+BASE_MBPS ([\d.]+)", re.M)
KQ_2_FINAL = re.compile(r"^FINAL S_KQ(\d+) ([\d.]+)\s+RAW_SLQP_MBPS ([\d.]+)\s+BASE_MBPS ([\d.]+)", re.M)
KQ_MQFT = re.compile(r"^KQ (\d+) MQFT ([\d.]+) SCORE ([\d.]+)$", re.M)


def read_text(name: str) -> str:
    raw = (LOGS / name).read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    return text.replace("\r\n", "\n")  # some original logs are UTF-16 with CRLF line endings


def parse(name: str, dataset: str, system: str) -> tuple[dict, float]:
    text = read_text(name)
    rx = KQ_2_FINAL if (system == "2cell" and dataset == "final") else KQ_7
    per_kq = {int(k): (float(s), float(base)) for k, s, _, base in rx.findall(text)}
    if system == "7cell":
        printed = re.search(r"^DIRECTOR_HELDOUTSCORE ([\d.]+)", text, re.M).group(1)
    elif dataset == "final":
        printed = re.search(r"^FINAL HELDOUTSCORE ([\d.]+)", text, re.M).group(1)
    else:
        printed = re.findall(r"^HELDOUTSCORE ([\d.]+)", text, re.M)[-1]
    return per_kq, float(printed)


def main() -> int:
    ok, kq, score = True, {}, {}
    print(f"{'Model':32s} {'Set':10s} {'Score (logged)':>15s} {'Mean of per-Kq':>15s}  Check")
    for key, label, name, dataset, system in RUNS:
        per_kq, printed = parse(name, dataset, system)
        n = 35 if system == "7cell" else 8
        mean = sum(s for s, _ in per_kq.values()) / len(per_kq)
        good = sorted(per_kq) == list(range(1, n + 1)) and abs(mean - printed) <= 2e-6  # per-Kq printed to 6 dp
        ok &= good
        kq[key], score[key] = per_kq, printed
        print(f"{label:32s} {dataset:10s} {printed!s:>15} {mean:15.6f}  {'OK' if good else 'FAIL'}")

    same = len({tuple(b for _, (_, b) in sorted(kq[k].items())) for k in ("A_fin", "B0_fin", "B2_fin")}) == 1
    ok &= same
    print(f"\nPath A, B0 and B2 final tests used identical channels (35/35 baseline values equal): {'YES' if same else 'NO'}")

    a, b0, b2 = score["A_fin"], score["B0_fin"], score["B2_fin"]
    wins = lambda x, y: sum(kq[x][k][0] > kq[y][k][0] for k in range(1, 36))
    print("\nProtected 7-cell final test:")
    print(f"  gap to Path A:   B0 {100 * (a - b0) / a:.2f} %   B2 {100 * (a - b2) / a:.2f} %")
    print(f"  B2 - B0 = {b2 - b0:+.6f}  (B2 higher at {wins('B2_fin', 'B0_fin')}/35 Kq; "
          f"recovers {100 * (b2 - b0) / (a - b0):.1f} % of B0's gap)")
    print(f"  A - B2  = {a - b2:+.6f}  (Path A higher at {wins('A_fin', 'B2_fin')}/35 Kq)")
    grid = sum(kq["A_fin"][k][0] for k in MQFT_GRID) / len(MQFT_GRID)
    print(f"  Path A mean over Kq {MQFT_GRID}: {grid:.6f}  (7-cell MQFT reference: reported in the paper, not in this repository)")

    mq = read_text("2cell_mqft_final_test.log")
    mq_score = float(re.search(r"^MQFT FINAL HELDOUTSCORE ([\d.]+)", mq, re.M).group(1))
    mq_kq = {int(k): float(s) for k, _, s in KQ_MQFT.findall(mq)}
    good = sorted(mq_kq) == list(range(1, 9)) and abs(sum(mq_kq.values()) / 8 - mq_score) <= 2e-6
    ok &= good
    print(f"\n2-cell final test: champion {score['2c_fin']:.6f}, MQFT reference {mq_score:.6f} "
          f"[{'OK' if good else 'FAIL'}] -> {100 * score['2c_fin'] / mq_score:.2f} % of MQFT")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
