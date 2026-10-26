# Discover Small, Deploy Large: Transferable Autoresearch for MU-MIMO Beamforming

In interference-limited multicell multiuser multiple-input multiple-output (MU-MIMO) networks, beamforming plays a central role in balancing aggregate throughput and cell-edge performance. Percentile-based beamforming provides a flexible framework for this tradeoff by optimizing the aggregate rate of the weakest fraction of users. However, the resulting optimization problem is non-convex and non-smooth, motivating the development of efficient, reusable beamforming policies.

Recent advances in agentic AI have enabled autonomous algorithm discovery through autoresearch, where AI agents autonomously discover and refine algorithmic solutions within a controlled experimental framework. An important question, however, remains: **Can an algorithm autonomously discovered for a smaller network be transferred to a larger network without repeating the discovery process?**

This repository accompanies our paper, which investigates **transferable autoresearch** for percentile-based multicell MU-MIMO beamforming through two experimental paths:

| Path | Steps | Folder |
|---|---|---|
| **A** | autoresearch directly on 7 cells | [`path_a/`](path_a/) |
| **B** | autoresearch on 2 cells → zero-shot transfer (**B0**) → conventional fine-tuning (**B2**) | [`path_b/`](path_b/) |

The independent campaigns discovered distinct communication-aware beamforming solutions: an interference-aware graph neural network (GNN) at the 2-cell scale and a Transformer-based architecture at the 7-cell scale.

The repository provides the discovered implementations, trained checkpoints, frozen evaluators, experimental records, and verification tools supporting the reported findings.

### → [`verify/`](verify/) — Verify reported results

```bash
python verify/verify_results.py
```

This script recomputes the reported results from the released evaluator logs without rerunning the experiments. The external 7-cell MQFT benchmark is excluded. It requires only the Python standard library and ends with `ALL CHECKS PASSED` (full output: `verify/expected_output.txt`).

### → [`path_a/`](path_a/) and [`path_b/`](path_b/) — inspect and re-run

Each folder contains the discovered beamforming implementations, trained checkpoints, experiment logs, and instructions for re-evaluation or retraining.

## Key Results

We evaluate all three policies on the same protected-final 7-cell evaluation set, comprising 250 channel realizations and all 35 percentile targets. Performance is measured using the normalized SLqP score, with an equal-power SVD baseline of 1.0.

| Policy | Target-side training | Final score | Gap to Path A |
|---|---:|---:|---:|
| **Path A:** Direct 7-cell Autoresearch | 8,766 s | **1.685** | — |
| **B0:** Zero-shot transfer | 0 s | **1.595** | 5.3% |
| **B2:** Transferred + fine-tuned | 474 s | **1.675** | 0.62% |

Fine-tuning improves B0 across all 35 percentile targets and recovers approximately **88.4%** of its performance gap to Path A. Path A performs better for K<sub>q</sub> = 1–25, while B2 performs slightly better for K<sub>q</sub> = 26–35.

Once the 2-cell champion is available, fine-tuning achieves a **94.6% reduction in target-side training time** compared with direct 7-cell Autoresearch, excluding the initial 2-cell discovery cost.

### Comparison with MQFT

Both autonomously discovered champions approach the performance of the multidimensional quadratic fractional transform (MQFT), a strong iterative optimization benchmark.

| Network | Autoresearch | MQFT | Ratio to MQFT |
|---|---:|---:|---:|
| 2-cell discovery network | 1.658 | 1.666 | **99.55%** |
| 7-cell network (six percentile targets) | 1.921 | 1.927 | **99.65%** |

The 7-cell MQFT comparison uses K<sub>q</sub> ∈ {1, 2, 4, 9, 18, 35}. Its MQFT reference value is reported in the paper but cannot be independently reproduced from this repository, since the corresponding MQFT records are not included. The percentages above are calculated from the original full-precision scores.

The search trajectories for both campaigns (Fig. 3 of the paper) and their underlying data are available in [`figures/`](figures/).

## Network Configurations

We consider two multicell MU-MIMO network configurations for autonomous discovery and cross-scale transfer.

| Parameter | 2-cell (Discovery) | 7-cell (Target) |
|---|---|---|
| Topology | Two adjacent hexagonal cells, no wraparound | Seven hexagonal cells with toroidal wraparound |
| Users per cell | 4 | 5 |
| Total users | 8 | 35 |

**Shared system assumptions:** Both networks employ 8 transmit antennas per base station (BS), 2 receive antennas per user, one data stream per user, a 43-dBm per-BS power constraint, Rayleigh fading, perfect channel-state information (CSI), and analytical minimum mean-squared error (MMSE) receive processing.

**Evaluation note:** The two network configurations use separate frozen evaluators. Consequently, their absolute normalized SLqP scores should not be directly compared across scales.

## Frozen Evaluators and Evaluation Integrity

To ensure reliable and reproducible evaluation, all candidates were assessed using frozen, SHA-256-pinned evaluators ([`evaluator/`](evaluator/)). The autonomous agent was permitted to modify only `train.py`.

For the 7-cell network, `director_rescore.py` separates candidate execution from authoritative evaluation. **Process A** executes the candidate code to generate beamformers, while **Process B** independently regenerates the channels and computes the scores without importing candidate code.

The integrity of the frozen evaluator files can be checked using:

```bash
sha256sum evaluator/7cell/prepare7.py evaluator/7cell/director_rescore.py
# 429ebd699f1b6f8ae6efb71bb73bd6c1c8002467b001fada7a515a978a21ca12  evaluator/7cell/prepare7.py
# a61f31a62f4632c5843faf567e1fa680502ffbeb4aabc619d2f8c8ae9e95b516  evaluator/7cell/director_rescore.py
```

### Model Selection and Protected-Final Evaluation

To prevent evaluation leakage, model selection and final performance assessment use separate, fixed channel sets.

- **Selection set:** Contains 250 channel realizations generated using a fixed, published seed (123). This set guides all keep/revert and model-selection decisions. Path A's champion was selected from 24 adaptive evaluations, while B0 and B2 each underwent one selection evaluation. These scores can be independently recomputed using the released evaluator.

- **Protected-final set:** Contains 250 independent channel realizations, with the seed withheld from the public repository. Each frozen champion was evaluated exactly once, after model selection was complete. These held-out evaluations provide the final performance estimates reported in the paper. The original results are available in [`verify/logs/`](verify/logs/), but reproducing the evaluations requires the unpublished seed.

## Reproducibility and Verification

The repository provides tools to verify the reported results and instructions for re-evaluating or retraining the discovered beamforming policies. The verification status of each component is summarized below.

| Component | Verification method | Status |
|---|---|---|
| Reported selection and final-test scores, performance gaps, and per-\(K_q\) comparisons | Recomputed from released evaluator logs using `verify/verify_results.py` | **Verified from logs** |
| Selection-set evaluation of Path A, B0, and B2 | Frozen evaluator and released checkpoints | Commands provided; not re-executed |
| Training of Path A, the 2-cell champion, and B2 | Training scripts in the respective campaign directories | Commands provided; bit-exact reproduction not guaranteed |
| Protected-final evaluations | Original evaluator logs | Results available; evaluation seed not released |

**Note:** Verification from logs confirms consistency with the recorded experimental results but does not constitute independent re-execution of the experiments.

## Computational Environment

Experiments were conducted on a Windows 11 system with an Intel Core i7-14700 CPU (20 PyTorch threads), Python 3.12.10, PyTorch 2.14.1+cpu, and NumPy 2.5.1.

Dependencies can be installed using:

`pip install -r requirements.txt`

The frozen computational limits (600 s per training run and 25 s per 7-cell evaluation) were calibrated for this hardware. Consequently, slower systems may exceed these limits.

## AI Agent

Both autonomous discovery campaigns, mechanical zero-shot transfer, and execution of the predefined fine-tuning procedure used Anthropic's Claude Code with Claude Opus 5.5 (`claude-opus-5-5`).

The human experimenter established the frozen research charters and evaluators, authorized each experimental stage, and conducted all protected-final evaluations.

The framework could, in principle, operate with other agentic AI tools offering comparable capabilities, although cross-agent generality has not been experimentally evaluated.

## Citing

```bibtex
@inproceedings{azadeginaeini2027discover,
  author    = {Azadegi Naeini, Sara and Adve, Raviraj and Khan, Ahmad Ali and bin Sediq, Akram},
  title     = {Discover Small, Deploy Large: Transferable Autoresearch for {MU-MIMO} Beamforming},
  booktitle = {Proc. IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2027}
}
```

## License
