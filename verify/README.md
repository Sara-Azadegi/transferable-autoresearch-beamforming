# verify/

| File | Contents |
|---|---|
| `verify_results.py` | Recomputes every reported number from `logs/`. Standard library only |
| `expected_output.txt` | Output of `python verify/verify_results.py` |
| `logs/` | The original evaluator outputs: selection and protected final test for the 2-cell champion, Path A, B0 and B2, plus the 2-cell MQFT reference. Several are UTF-16 text. The directory part of local file-system paths (temporary-file, lock-file and checkpoint locations) was replaced by `<LOCAL_PATH>`; no other character was changed |
| `results/main_results.csv` | One row per model: selection score, final-test score, inference time |
| `results/per_kq_7cell_final_test.csv` | Path A, B0 and B2 final-test scores per K<sub>q</sub>, with differences and the shared-baseline check |
| `results/per_kq_2cell_final_test.csv` | 2-cell champion final-test scores per K<sub>q</sub> |

`../figures/` holds Fig. 3 of the paper (selection-set search trajectories) and the CSV data it
plots.
