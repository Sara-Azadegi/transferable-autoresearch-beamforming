# Path B — Transferable Autoresearch: Discover Small, Deploy Large

Path B investigates whether a beamforming algorithm autonomously discovered in a smaller **2-cell network** can be transferred to a **7-cell wraparound network** without repeating autonomous discovery.

The experimental workflow consists of three stages:

| Stage | Description | Folder | 7-cell training |
|---|---|---|---:|
| **Discovery** | Autonomous algorithm discovery on the 2-cell network | [`1_discovery_2cell/`](1_discovery_2cell/) | — |
| **B0: Zero-shot transfer** | Mechanical transfer of the frozen 2-cell champion to 7 cells | [`2_zero_shot_B0/`](2_zero_shot_B0/) | 0 s |
| **B2: Fine-tuning** | Conventional parameter adaptation using a pre-registered procedure, without repeating Autoresearch | [`3_finetune_B2/`](3_finetune_B2/) | 474 s |

## 1. Autonomous Discovery on the 2-Cell Network

The discovery campaign operates on two adjacent non-wraparound cells, each serving four users, and optimizes the percentile-beamforming objective across all eight percentile targets.

Under the frozen research charter (`campaign/program.md`) and evaluator (`campaign/prepare.py`), the autonomous agent explores beamforming algorithms through controlled experiments.

### Discovered Architecture

The campaign produced an **interference-aware graph neural network (GNN)**, combining:

- User-centric message passing with separate same-cell and inter-cell aggregation.
- A structured RZF/SLNR-like beamformer.
- MMSE-based transmit/receive refinement.

**Selected champion:** `InterferenceGNN` (Run-25), with 984,006 trainable parameters, stored in `campaign/baby_champion_run25.pt`.

### Campaign Summary

| Metric | Recorded result |
|---|---|
| Network | 2 cells, 8 users |
| Recorded runs | 26, including the initial seed |
| Experiment outcomes | 17 COMMIT, 8 REVERT, 1 FAIL |
| Commit threshold | 0.000289 |
| Cumulative training time | 10,315 s |
| Model-selection score | 1.644 |
| Protected-final score | **1.658** |
| MQFT reference | 1.666 |

The champion achieves **99.55% of the MQFT reference score** on the protected-final evaluation set.

The implementation and experimental changelog are preserved in `campaign/train.py`, while `log.csv` records the campaign history.

**Retraining command** (from `path_b/`):

```bash
cd 1_discovery_2cell/campaign
python train.py
```

This executes the archived Run-25 training procedure and writes `last_model.pt`. The command was not re-executed during repository preparation.

## 2. Zero-Shot Transfer to the 7-Cell Network (B0)

The frozen 2-cell champion is transferred to the 7-cell network **without target-side training, optimizer updates, or changes to its 62 learned parameter tensors**.

Only mechanical compatibility adaptations are introduced, documented in `transfer_Z1-Z4.diff`:

| Step | Adaptation |
|---|---|
| Z1 | Adaptation to the frozen 7-cell evaluator interface |
| Z2 | Generalization of the non-serving BS feature to the summed effective gain from all non-serving BSs, reducing to the original feature for two cells |
| Z3 | Strict checkpoint loading through `load_candidate` |
| Z4 | Inference-speed optimization, with a verified maximum relative beam difference of \(1.15\times10^{-5}\) |

The discovered GNN architecture and learned weights remain unchanged. The transfer does not use 7-cell performance feedback to redesign the algorithm.

The `evaluation_package/` directory contains the evaluated B0 implementation and frozen evaluator files.

### Re-Evaluation

From `path_b/`:

```bash
cd 2_zero_shot_B0/evaluation_package
python director_rescore.py ../../1_discovery_2cell/campaign/baby_champion_run25.pt --hash-file ../../../evaluator/7cell/DIRECTOR_HASHES.sha256
```

Expected selection-set output:

```text
DIRECTOR_HELDOUTSCORE 1.595487202664
```

| Evaluation | Normalized SLqP score |
|---|---:|
| Model-selection score | 1.595 |
| Protected-final score | **1.595** |

The command was not re-executed during repository preparation.

## 3. Conventional Fine-Tuning on the 7-Cell Network (B2)

Following zero-shot evaluation, the transferred champion undergoes **conventional parameter fine-tuning**, retaining its discovered architecture, feature representation, beamformer construction, and inference procedure.

The fine-tuning recipe was fixed before execution and did not involve a new autonomous algorithm-discovery campaign.

### Fine-Tuning Configuration

| Parameter | Configuration |
|---|---|
| Initialization | Frozen 2-cell champion (`inputs/baby_run25.pt`) |
| Optimization | Adam |
| Training updates | 1,000 |
| Training channels per update | 64 fresh 7-cell realizations |
| Training stream seed | 789 |
| Peak learning rate | \(2\times10^{-4}\) |
| Learning-rate schedule | 100-update linear warm-up followed by cosine decay |
| Gradient clipping | None |
| Architecture | Unchanged |
| Recorded training time | 474 s |

The checkpoint initialization hash is verified by the training code, and the recorded run output is available in `train_stdout.log`.

### Training and Re-Evaluation

From `path_b/`:

```bash
cd 3_finetune_B2/campaign

# Execute the archived fine-tuning procedure
python -B train.py

# Re-evaluate the newly trained model
python director_rescore.py last_model.pt --hash-file ../../../evaluator/7cell/DIRECTOR_HASHES.sha256

# Re-evaluate the released B2 champion
python director_rescore.py ../s2ft1_champion.pt --hash-file ../../../evaluator/7cell/DIRECTOR_HASHES.sha256
```

Expected selection-set output for the released B2 champion:

```text
DIRECTOR_HELDOUTSCORE 1.676352664207
```

| Evaluation | Normalized SLqP score |
|---|---:|
| Model-selection score | 1.676 |
| Protected-final score | **1.675** |

Fine-tuning improves zero-shot performance across all 35 percentile targets and brings the transferred champion within **0.62% of independent 7-cell Autoresearch (Path A)**.

### Checkpoint Integrity

The released `s2ft1_champion.pt` is a sanitized copy of the evaluated B2 checkpoint.

Only the local path stored in the `init_checkpoint` metadata field was changed. **All 62 learned parameter tensors remain byte-identical** to the original evaluated checkpoint.

The original and release-copy SHA-256 hashes differ because of this metadata modification. The verification record is retained separately from the public repository.

## Reproducibility Notes

- Selection-set re-evaluation commands are provided but were not re-executed during repository preparation.
- Retraining is not guaranteed to produce bit-identical results because deterministic algorithms were not enforced.
- Protected-final scores are supported by the original evaluator logs; the protected evaluation seed is not released.
- Numerical results can be independently checked against the released logs using [`../verify/`](../verify/).

Absolute scores from the 2-cell and 7-cell evaluators should not be compared directly because they correspond to different network configurations.
