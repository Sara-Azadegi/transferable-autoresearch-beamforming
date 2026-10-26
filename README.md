# Discover Small, Deploy Large: Transferable Autoresearch for MU-MIMO Beamforming

An AI coding agent, working under a frozen evaluator and a research charter, discovered a
percentile-beamforming algorithm on a **2-cell** network. We deployed that discovery **zero-shot**
on a **7-cell wraparound** network, with no 7-cell training and every learned parameter
unchanged. We then **fine-tuned** it conventionally. The result is compared with a separate
autoresearch campaign run directly on the 7-cell network.

> S. Azadegi Naeini, R. Adve, A. A. Khan, and A. bin Sediq, "Discover Small, Deploy Large:
> Transferable Autoresearch for MU-MIMO Beamforming," *IEEE WCNC*, 2027.

| Path | Steps | Folder |
|---|---|---|
| **A** | autoresearch directly on 7 cells | [`path_a/`](path_a/) |
| **B** | autoresearch on 2 cells → zero-shot transfer (**B0**) → conventional fine-tuning (**B2**) | [`path_b/`](path_b/) |

### → [`verify/`](verify/) — check the reported numbers

```bash
python verify/verify_results.py
```

This recomputes every score below from the original evaluator logs. It needs only the Python
standard library and ends with `ALL CHECKS PASSED` (full output: `verify/expected_output.txt`).

### → [`path_a/`](path_a/) and [`path_b/`](path_b/) — inspect and re-run

Each folder has the champion checkpoints, the code that produced them, the campaign log and
the commands to re-score or retrain.

## The result in one table

Protected 7-cell final test: 250 unseen channels, the **same channels for all three policies**,
K<sub>q</sub> = 1..35. Normalized SLqP score: the equal-power SVD baseline is 1.0.

| Policy | 7-cell training | Final-test score | Gap to Path A |
|---|---:|---:|---:|
| Path A: direct 7-cell autoresearch (Run-18) | 8,766 s | **1.684942** | — |
| B0: zero-shot transfer of the 2-cell champion | 0 s | **1.595309** | 5.32 % |
| B2: B0 + conventional fine-tuning | 474 s | **1.674505** | 0.62 % |

* Fine-tuning improves B0 at all 35 K<sub>q</sub> and recovers 88.4 % of its gap to Path A.
  Path A is higher at K<sub>q</sub> = 1–25 and B2 at K<sub>q</sub> = 26–35.
* **Cost:** including the 2-cell campaign (10,315 s), Path B used 10,789 s of training in
  total, 23 % more than Path A. Transfer saves training on the **target** network, not in total.
* **MQFT reference:**
  * 2-cell final test: the 2-cell champion scores 1.658049 against MQFT's 1.665520
    (**99.55 %**). Both values are in `verify/logs/`.
  * 7-cell, K<sub>q</sub> ∈ {1, 2, 4, 9, 18, 35}: Path A scores 1.920759, which can be
    verified here. The paper reports MQFT at 1.927 (99.65 %). The MQFT run itself is not part
    of this repository, so that value is cited from the paper, not reproduced here.

The search trajectories of both campaigns (Fig. 3 of the paper) and the data they plot are in
[`figures/`](figures/).

## The two networks

| | 2-cell (discovery) | 7-cell (target) |
|---|---|---|
| Topology | 2 adjacent hexagonal cells, no wraparound | 7 hexagonal cells, toroidal wraparound |
| Users | 4 per cell (K = 8) | 5 per cell (K = 35) |
| Common | M = 8 BS antennas, N = 2 receive antennas, one stream per user, 43 dBm per BS, Rayleigh fading, perfect CSI, MMSE receivers | same |

Scores from the two networks are not comparable with each other.

## The evaluator is the whole argument

Every candidate was scored by evaluators that were frozen and SHA-256 pinned before the first
experiment ([`evaluator/`](evaluator/)). The agent could change only `train.py`. In the 7-cell
director (`director_rescore.py`), Process A runs the candidate's code to produce beamformers.
Process B, which never imports that code, regenerates the channels and computes the score.
Check your copy:

```bash
sha256sum evaluator/7cell/prepare7.py evaluator/7cell/director_rescore.py
# 429ebd699f1b6f8ae6efb71bb73bd6c1c8002467b001fada7a515a978a21ca12  evaluator/7cell/prepare7.py
# a61f31a62f4632c5843faf567e1fa680502ffbeb4aabc619d2f8c8ae9e95b516  evaluator/7cell/director_rescore.py
```

**Selection vs. protected final test.**

* The *selection set* (250 channels, seed 123, built into the evaluator) drove every keep/revert
  and model-selection decision. Path A's selection score is the best of 24 adaptive
  evaluations; B0 and B2 had one each. Anyone can re-score on it.
* The *protected final set* (250 channels from a seed held by the experimenter and not
  published) was used **once** per frozen champion, after selection was over. Only these scores
  are unbiased; the final tests cannot be re-run, and their original logs are in `verify/logs/`.

## What can be reproduced from this repository

| Claim | How | Status |
|---|---|---|
| All final-test and selection scores, gaps and per-K<sub>q</sub> comparisons | `verify/verify_results.py` (recomputed from the original evaluator logs; no model is run) | checked for this release |
| Selection-set scores of Path A, B0 and B2 | frozen director on the released checkpoints (`path_a/`, `path_b/`) | commands provided; not re-run for this release |
| Training of Path A Run-18, the 2-cell champion and B2 | `train.py` in each campaign folder | commands provided; not re-run; not expected to be bit-exact (deterministic algorithms were not enforced) |
| Protected final-test scores | needs the unpublished seed | logs only |
| 7-cell MQFT reference (1.927) | — | reported in the paper; not in this repository |

## Environment

Intel Core i7-14700 (CPU only, 20 PyTorch threads), Windows 11, Python 3.12.10,
PyTorch 2.14.1+cpu and NumPy 2.5.1 (`pip install -r requirements.txt`). The frozen time limits
(600 s per training run, 25 s per 7-cell evaluation) were set for this machine, so slower
hardware may fail them.

## AI agent

Both discovery campaigns, the zero-shot adaptation and the execution of the fixed fine-tuning
recipe were carried out with Anthropic's Claude Code using Claude Opus 5.5
(`claude-opus-5-5`). The human experimenter froze the charters and evaluators, approved every
stage and ran every protected final test. Other agentic tools could in principle drive the same
framework; this has not been tested.

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

To be added.
