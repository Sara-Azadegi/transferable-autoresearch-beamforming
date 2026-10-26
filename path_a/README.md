# Path A: direct 7-cell autoresearch

An AI agent ran autoresearch directly on the 7-cell wraparound network (35 users,
K<sub>q</sub> = 1..35) under the charter `campaign/program.md`. It started from a generic MLP
seed, written and run by the experimenter, and received nothing from the 2-cell campaign.

| File | Contents |
|---|---|
| `campaign/champion_run18.pt` | **Champion (Run-18)**: `UserTransformerDualBF4StageRelMu`, 275,106 parameters. A user-token Transformer with pairwise interference attention biases, a KKT-inspired dual-variable beamformer and three MMSE-aware refinement stages |
| `campaign/train.py` | Champion source, byte-identical in code to the Run-18 commit. It ends with the agent's append-only changelog: hypothesis, falsifier, result and decision for each experiment, and the early-stop rationale |
| `campaign/prepare7.py`, `director_rescore.py`, `program.md`, `requirements.txt` | Frozen evaluator files (identical to `../evaluator/7cell/`). The director requires them next to `train.py` |
| `log.csv` | All 24 runs: family, commit, training time, selection score, per-K<sub>q</sub> scores and COMMIT/REVERT status |

**Campaign:**

* Seed run 0 plus 23 agent-launched experiments, out of a cap of 50: 15 COMMIT and 9 REVERT.
* Three architecture families.
* Commit rule: S<sub>new</sub> > S<sub>best</sub> + 0.0005.
* 8,766.3 s of training in total; stopped early after 5 consecutive non-commits.

**Scores:** selection 1.686715818726 (best of 24); protected final test **1.684941922076**.

## Commands (run from `path_a/campaign/`)

```bash
# Re-score the champion on the selection set. Expected last line:
# DIRECTOR_HELDOUTSCORE 1.686715818726
python director_rescore.py champion_run18.pt --hash-file ../../evaluator/7cell/DIRECTOR_HASHES.sha256

# Retrain with the Run-18 recipe (seeds 2027/789; must finish within 600 s). Writes last_model.pt.
python train.py
```

These commands were not re-run when this repository was prepared.
