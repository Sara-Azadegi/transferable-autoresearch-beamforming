# Path B: discover small (2 cells), deploy large (7 cells)

| Step | Folder | Kind of work | 7-cell training |
|---|---|---|---:|
| 1 | `1_discovery_2cell/` | **Autonomous discovery** by the AI agent on the 2-cell network | — |
| 2 | `2_zero_shot_B0/` | **Mechanical transfer** of the frozen 2-cell champion | 0 s |
| 3 | `3_finetune_B2/` | **Conventional fine-tuning** with one pre-registered recipe; no agent-driven search | 474 s |

## 1. 2-cell discovery

* **Network:** 2 cells, 4 users per cell, K<sub>q</sub> = 1..8, no wraparound.
* **Charter and evaluator:** `campaign/program.md` and `campaign/prepare.py`, identical to
  `../evaluator/2cell/`.
* **Campaign:** seed run 0 plus 25 agent-launched experiments: 17 COMMIT, 8 REVERT and 1 FAIL
  (the training time limit). Commit rule δ = 0.000289. 10,315 s of training in total.
* **Champion:** `campaign/baby_champion_run25.pt`, an `InterferenceGNN` with 984,006
  parameters: a user-centric GNN with separate same-cell and cross-cell message passing, a
  structured RZF/SLNR-like beamformer, and MMSE-based transmit/receive refinement. Its source is
  `campaign/train.py`, which includes the agent's changelog. `log.csv` lists all 26 runs.
* **Scores (2-cell system):** selection 1.644191; protected final test **1.658049**
  (MQFT reference 1.665520).
* **Retraining:** `cd 1_discovery_2cell/campaign && python train.py` re-runs the Run-25 recipe
  and writes `last_model.pt`. Not re-run for this release.

## 2. Zero-shot transfer (B0)

The 2-cell champion runs on the 7-cell network with **all 62 learned tensors and the
architecture unchanged**: 0 optimizer steps and no 7-cell data. Only mechanical adaptations
were made (`transfer_Z1-Z4.diff`):

* **Z1:** the 7-cell evaluator API.
* **Z2:** the "other BS" feature becomes the summed effective gain from all non-serving BSs
  (identical to the original feature when B = 2).
* **Z3:** a strict `load_candidate` replaces the training driver.
* **Z4:** an inference-speed rewrite (maximum relative beam difference 1.15·10⁻⁵).

`evaluation_package/` is the exact package that was evaluated: the frozen files plus `train.py`.

```bash
cd 2_zero_shot_B0/evaluation_package
python director_rescore.py ../../1_discovery_2cell/campaign/baby_champion_run25.pt --hash-file ../../../evaluator/7cell/DIRECTOR_HASHES.sha256
# expected: DIRECTOR_HELDOUTSCORE 1.595487202664
```

**Scores:** selection 1.595487202664; protected final test **1.595308837276**.
The command was not re-run for this release.

## 3. Conventional fine-tuning (B2)

The fine-tuning recipe was fixed before the single run:

* initialization `inputs/baby_run25.pt` (the 2-cell champion; its hash is checked in code);
* 1000 updates × 64 fresh 7-cell training channels (stream seed 789);
* Adam with peak learning rate 2·10⁻⁴, 100-update linear warm-up, then cosine decay;
* no clipping; architecture unchanged.

The recorded run took 474.09 s (`train_stdout.log`).

```bash
cd 3_finetune_B2/campaign
python -B train.py        # writes last_model.pt (refuses to overwrite)
python director_rescore.py last_model.pt --hash-file ../../../evaluator/7cell/DIRECTOR_HASHES.sha256
# Released checkpoint; expected DIRECTOR_HELDOUTSCORE 1.676352664207:
python director_rescore.py ../s2ft1_champion.pt --hash-file ../../../evaluator/7cell/DIRECTOR_HASHES.sha256
```

**Scores:** selection 1.676352664207; protected final test **1.674505154990**.
None of these commands were re-run for this release, and retraining is not expected to be
bit-exact.

**About `s2ft1_champion.pt`:** this is a release copy of the evaluated checkpoint (SHA-256
`3e323b13…c2fc9`). A local file path was removed from its `init_checkpoint` metadata field.
All 62 weight tensors are byte-identical to the evaluated checkpoint; the release copy's
SHA-256 is `c62c0754…c53f`.
