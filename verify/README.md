# Results Verification

This directory contains the experimental logs, numerical results, and verification tools supporting the findings reported in our paper.

The verification script recomputes reported performance metrics from the released evaluator logs **without rerunning training or evaluation**.

## Running Verification

From the repository root, execute:

```bash
python verify/verify_results.py
```

The script requires only the Python standard library. Successful verification ends with `ALL CHECKS PASSED`. The reference output is provided in [`expected_output.txt`](expected_output.txt).

## Directory Contents

| File | Description |
|---|---|
| [`verify_results.py`](verify_results.py) | Recomputes reported scores and performance comparisons from the available evaluator logs. |
| [`expected_output.txt`](expected_output.txt) | Reference output of the verification script. |
| `logs/` | Evaluator logs for selection and protected-final evaluations of the 2-cell champion, Path A, B0, and B2, along with the 2-cell MQFT reference. |
| [`results/main_results.csv`](results/main_results.csv) | Model-level selection scores, protected-final scores, and inference times. |
| [`results/per_kq_7cell_final_test.csv`](results/per_kq_7cell_final_test.csv) | Per-percentile protected-final scores for Path A, B0, and B2, including performance differences and the shared-baseline check. |
| [`results/per_kq_2cell_final_test.csv`](results/per_kq_2cell_final_test.csv) | Per-percentile protected-final scores for the 2-cell champion. |

## Important Notes

Some evaluator logs use UTF-16 encoding. Local directory paths have been replaced with `<LOCAL_PATH>` to protect personal information while preserving the numerical results.

Verification confirms consistency with the recorded experimental results; it does not constitute independent re-execution of the experiments. The external 7-cell MQFT reference is not included in these logs.

The autoresearch search trajectories (Fig. 3 of the paper) and their underlying CSV data are available in [`figures/`](../figures/).
