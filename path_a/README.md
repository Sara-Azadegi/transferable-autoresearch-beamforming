# Path A — Direct 7-Cell Autoresearch

Path A investigates **independent autonomous beamforming algorithm discovery** directly on a 7-cell wraparound MU-MIMO network comprising 35 users. The campaign optimizes the percentile-based beamforming objective across all 35 percentile targets.

Starting from a generic multilayer perceptron (MLP) candidate, the AI agent autonomously explored algorithmic designs under a frozen evaluation protocol and research charter. Crucially, **Path A received no architecture, learned parameters, features, or experimental information from the 2-cell campaign**, ensuring an independent target-scale comparison.

## Discovered Beamforming Architecture

The campaign produced a **Transformer-based beamforming policy** combining learned relational processing with communication-aware beamformer construction.

Its main components include:

- **User-token Transformer:** Represents users using physically motivated channel and interference features.
- **Interference-aware attention:** Incorporates pairwise interference information into attention biases.
- **KKT-inspired beamformer:** Predicts power-related variables, including per-BS multipliers and user-dependent interference weights.
- **MMSE-aware refinement:** Applies three additional receive-aware stages, yielding a four-stage beamforming policy.

**Selected champion:** `UserTransformerDualBF4StageRelMu` (Run-18), with 275,106 trainable parameters.

## Campaign Summary

| Metric | Recorded result |
|---|---|
| Network | 7-cell wraparound, 35 users |
| Initial candidate | Generic MLP |
| Launched runs | 24, including the initial seed |
| Architecture families | 3 |
| Maximum permitted experiments | 50 |
| Commit threshold | \(S_{\mathrm{new}} > S_{\mathrm{best}} + 0.0005\) |
| Cumulative candidate-training time | 8,766 s |
| Stopping criterion | Approved early stop after five consecutive non-improving runs |

The campaign retained a permanent record of experimental hypotheses, outcomes, and commit/revert decisions.

## Performance

| Evaluation | Normalized SLqP score |
|---|---:|
| Model-selection score | 1.687 |
| Protected-final score | **1.685** |

The protected-final score was obtained from 250 independent channel realizations after the champion was frozen. Model-selection and protected-final results are reported separately.

## Repository Contents

| File | Description |
|---|---|
| `campaign/champion_run18.pt` | Trained checkpoint of the selected Transformer champion |
| `campaign/train.py` | Run-18 implementation, training procedure, and append-only experimental changelog |
| `campaign/prepare7.py` | Frozen 7-cell system model and evaluation preparation |
| `campaign/director_rescore.py` | Independent evaluator for selection-set scoring |
| `campaign/program.md` | Frozen research charter governing autonomous experimentation |
| `campaign/requirements.txt` | Campaign-specific dependencies |
| `log.csv` | Complete campaign log containing experiment outcomes, scores, training times, and commit/revert decisions |

The frozen evaluator files are also available in [`../evaluator/7cell/`](../evaluator/7cell/).

## Re-Evaluation and Training

Run the following commands from `path_a/campaign/`.

**Re-evaluate the frozen champion on the selection set:**

```bash
python director_rescore.py champion_run18.pt --hash-file ../../evaluator/7cell/DIRECTOR_HASHES.sha256
```

Expected final output:

```text
DIRECTOR_HELDOUTSCORE 1.686715818726
```

**Run the archived Run-18 training procedure:**

```bash
python train.py
```

This training command uses the recorded seeds (2027/789), has a 600-second computational limit, and writes `last_model.pt`.

**Reproducibility note:** These commands were provided but were not re-executed during repository preparation. The protected-final evaluation cannot be independently rerun using the released files because its evaluation seed is not published. Recorded results can instead be checked through [`../verify/`](../verify/).
