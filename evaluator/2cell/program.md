# Baby Percentile Beamforming Autoresearch — Research Charter

> **PRE-CAMPAIGN FINALIZATION.** The physical problem, evaluator contract, initial
> `train.py` seed, calibration, and C4 autonomous-campaign protocol are frozen from
> the director decisions. The autonomous campaign MUST NOT begin until
> `prepare.py` and this file are SHA-256 pinned and the seed state is committed in Git.

## 1. Goal

Design a learned centralized beamforming policy for the frozen Baby Percentile
Beamforming v1.1 problem:

```python
V = model(H, Kq)
```

- `H`: complex `[batch, 4, 2, 2, 2, 8]`, with
  `H[t,k,b,bp,:,:] = H_{bp -> k,b}`.
- `Kq`: percentile count in `{1,...,8}`.
- `V`: complex `[batch, 4, 2, 8]`, with `V[t,k,b,:] = v_{k,b}`.

One parameter set must serve every `Kq=1,...,8`.

Theoretical reference: A. A. Khan and R. S. Adve, "Percentile Optimization in
Wireless Networks—Part II: Beamforming for Cell-Edge Throughput Maximization,"
IEEE Trans. Signal Process., vol. 72, 2024. "Part II" below refers to this paper.
The paper itself is NOT provided. Work only from this charter; do not search
for, download, or reconstruct its algorithms beyond what is stated here.

## 2. Frozen Baby-v1.1 physics — do not redefine

- 2 adjacent regular hexagonal cells; no wraparound.
- BS coordinates `(0,0)` m and `(2000,0)` m.
- Hexagon apothem `1000 m`, 30-degree-offset orientation.
- 4 simultaneously active users per cell; 8 users total; no scheduling.
- BS transmit antennas `M=8`; user receive antennas `N=2`.
- One data stream per user.
- Users are i.i.d. uniform over their serving hexagon via the exact six-triangle
  sampling method implemented in `prepare.py`.
- Rayleigh block fading with entries `CN(0,1)` implemented as
  `(X + jY)/sqrt(2)`.
- Path-loss amplitude:

  `L(d) = (1 + d/0.392)^(-3.76/2)`.

- No shadowing.
- Perfect instantaneous global CSI.
- Noise PSD `-143 dBm/Hz`; bandwidth `20 MHz`.
- Per-BS total transmit-power constraint, exactly as in Part II Problem (3b):

  `sum_{k=1}^4 ||v_{k,b}||^2 <= Pmax` for each BS `b`, with `Pmax = 43 dBm`.

  There is no separate per-user power cap.

- All physical calculations in the immutable evaluator use linear watts.

## 3. Frozen objective

For network rates `r` and `Kq in {1,...,8}`:

`f_Kq(r) = sum of the Kq smallest rates across all 8 users`.

The evaluation rate is `20*log2(1+SINR)` Mbps. The receive combiner is NOT a
learned output: `prepare.py` computes the analytical MMSE combiner

`u = R_yy^{-1} H_serv v_des`,

where `R_yy` contains receiver noise and all desired/interfering streams. The
SINR denominator excludes the desired stream.

The training Kq distribution lives in `train.py`; the seed uses Uniform{1,...,8}
and the agent MAY change it. The evaluation set Kq = 1,...,8 may NEVER change.

## 4. Immutable evaluator authority

`prepare.py` is the sole authority for:

- geometry / user drops / channel generation;
- all physical constants and unit conversion;
- selection and final benchmark generation;
- analytical MMSE receive processing;
- exact Mbps rates and exact hard SLqP;
- beamformer shape / finite-value / power feasibility checks;
- normalization baseline;
- inference timing and contract enforcement;
- the authoritative `HELDOUTSCORE`.

The evaluator uses `complex128 / float64`. Training may use other numerical
representations inside `train.py`.

`prepare.py` must NEVER clip, normalize, project, or repair an infeasible model
output. Invalid shape, NaN/Inf, power violation, state mutation at evaluation,
or inference-budget violation is `FAIL / no score`.

Power validation permits only the numerical guard

`sum_k ||v_{k,b}||^2 <= Pmax * (1 + 1e-6)` for each BS `b`.

MMSE is computed with `torch.linalg.solve` and the physical `sigma^2 I` only;
no artificial diagonal jitter is permitted in the immutable evaluator.

## 5. Selection metric

The pinned selection set contains 250 independently generated Baby-v1.1 channel
realizations from `SEED_SELECT=123`. The SAME 250 channels are evaluated for
all `Kq=1,...,8` (sample matching).

For each `Kq`:

`S_Kq = mean_H f_model,Kq(H) / mean_H f_baseline,Kq(H)`.

The single authoritative scalar is

`HELDOUTSCORE = (1/8) * sum_{Kq=1}^8 S_Kq`.

Only `HELDOUTSCORE` controls commit/revert. Per-Kq normalized scores, raw mean
SLqP Mbps, inference time, and validity checks are diagnostics.

Per-Kq scores `S_Kq` are logged for every run. If, at calibration, one Kq
(expected: Kq=1) dominates the mean, the director reviews the metric before
`prepare.py` is pinned.

## 6. Fixed normalization baseline

The score-1.000 floor is equal-power dominant-direct-channel-singular-vector
beamforming. For each serving channel

`H_serv = U Sigma V^H`,

use

`v_base = sqrt(Pmax / K_USERS) * v_max(H_serv)`.

With `K_USERS=4`, each stream receives `Pmax/4`, so each BS exactly uses the
shared budget `Pmax`. It is Kq-independent, feasible, and evaluated through the same analytical MMSE
receiver and exact SLqP evaluator.

## 7. Strong classical reference: MQFT (Part II, Algorithm 1)

MQFT is an OFFLINE comparison bar only. It may never be invoked from model
inference.

Frozen MQFT choices:

- same pinned channels and all `Kq=1,...,8`;
- initialization = the fixed equal-power (`Pmax/4` per stream) dominant-SVD baseline;
- relative objective tolerance `1e-6`;
- the V-update uses CLARABEL, falling back to SCS (as in the power-control
  reference); every objective decrease between iterations is counted and
  reported (`decreases_by_kq`); a nonzero count is flagged, not hidden.
- maximum `25` iterations;
- if iteration 25 is reached before tolerance, explicitly report
  `NOT CONVERGED`; never silently label it converged.

`prepare.py` contains the reference implementation path. It may require CVXPY
for the offline fixed-X convex subproblem; that dependency is not part of model
inference.

## 8. Training / selection / final separation

Three independent random streams are frozen:

- training channel stream: `SEED_TRAIN=789`, reset every complete experiment;
- selection set: `SEED_SELECT=123`;
- final set: seed held by the director outside the repository and read only
  from the environment variable `BABY_FINAL_SEED` at final evaluation.

The generator uses NumPy `default_rng(seed)`, with one sequential RNG stream:
positions first, then all real Rayleigh draws, then all imaginary Rayleigh draws.

Selection/final channels must NEVER enter training or fine-tuning.

The final set is inaccessible by protocol during the campaign. After the
champion is frozen, it is evaluated exactly once with the identical scoring
procedure. No model change, retraining, hyperparameter change, or retuning is
permitted after final results are revealed.

`train.py` may import from `prepare.py` ONLY: physical constants,
`TrainingChannelStream`, and `evaluate_selection`. It must NEVER access
`_SELECT_H`, `sample_channels` with seed 123, `evaluate_final`,
`run_mqft_reference`, or any name starting with `_`.

## 9. Inference contract

At evaluation, trained parameters are frozen. The only permitted operation is a
fixed-cost mapping

`(H, Kq) -> V`.

Forbidden at inference:

- gradient updates / fine-tuning;
- optimizing `V` separately for each held-out instance;
- restarts/candidate searches selected by measured utility;
- adaptive loops such as "iterate until convergence".

Deterministic algebraic post-processing and a fixed predeclared number of
iterations are allowed, provided the total inference-time contract is met.

The inference budget is fixed at `T_max = 5.0 s` for all 250 x 8 selection
tasks on the designated campaign CPU (timing excludes channel generation,
scoring, file I/O, and process startup). It is scaled from the power-control
campaign (10 s for 4250 tasks, ~2.4 ms/task). The seed's measured time
(`measure_inference_time`, median of repeats) is reported for reference only.
Exceeding `T_max` is `FAIL / no score`.

Designated campaign CPU: Intel Core i7-14700 (Windows), PyTorch 2.14.1+cpu,
20 intra-op threads (fixed with `torch.set_num_threads(20)`). All timing,
calibration, and campaign runs use this machine and configuration.

## 10. Commit/revert noise band

The initial seed `train.py` was run five times on the designated CPU, each
with a different seed pair set via `CALIB_TORCH_SEED` / `CALIB_TRAIN_SEED`:
(2027,789), (2028,790), (2029,791), (2030,792), (2031,793).
The measured HELDOUTSCORE values were
`0.998952, 0.998901, 0.998928, 0.999190, 0.998950`.
These overrides are for calibration only and must be unset during the campaign.
Therefore

`delta = max(S_i) - min(S_i) = 0.000289`.

A candidate may be committed only if

`S_new > S_champion + delta`.


The campaign uses fixed seeds and the designated campaign hardware, but does
NOT globally force `torch.use_deterministic_algorithms(True)`.

## 11. Source-file authority

Core research files are exactly:

- `prepare.py` — immutable evaluator;
- `train.py` — ONLY mutable research source file;
- `program.md` — immutable research charter.

No new `.py`, `.md`, notebook, or other source file may become an alternate
location for research logic. Runtime products such as `log.csv`, checkpoints,
diagnostics, and regenerable caches are allowed.

The authoritative narrative experiment changelog is append-only inside
`train.py`. Prior entries must not be rewritten or deleted.

Before unattended research begins, `prepare.py` and `program.md` must be
SHA-256 pinned and their frozen hashes recorded by the director.

## 12. Initial seed `train.py` — frozen starting point

The campaign starts from a deliberately plain but valid centralized learner:

- full global CSI is available to one centralized NN;
- input = `asinh(Re H / sigma)`, `asinh(Im H / sigma)`, the unit-norm baseline
  beams `v_base/sqrt(Pmax/4)` (Re, Im), and `Kq/8` = 641 real scalars, where
  `sigma = sqrt(NOISE_POWER_W)` and asinh is applied elementwise;
- MLP `641 -> 512 -> 512 -> 128`, ReLU hidden activations;
- 128 outputs represent a complex correction `z` (64 coefficients);
- output `V = V_base + sqrt(Pmax/4) * z`, where `V_base` is the §6 baseline
  computed inside the model; then each BS block is rescaled by
  `min(1, sqrt(Pmax / sum_k ||v_{k,b}||^2))`, so power is feasible and may be
  allocated unequally among the four users;
- the last layer is zero-initialized, so the untrained seed equals the
  baseline (HELDOUTSCORE ≈ 1.000);

- differentiable training copy of Baby MMSE/SINR/rate physics lives inside
  `train.py`; `prepare.py` remains the independent judge;
- exact hard SLqP via sort + cumulative sum; no smoothing or surrogate;
- loss = negative raw SLqP averaged over batch;
- Adam, constant learning rate `1e-3`;
- seed default: exactly 2000 optimizer updates;
- batch = 128 fresh channel realizations each update;
- independent `Kq ~ Uniform{1,...,8}` per training sample;
- network in `float32`; training physics (MMSE/SINR/rate) in
  `float64 / complex128` to avoid cancellation NaNs for users near the BS;
- standard PyTorch `nn.Linear` initialization, except the last layer (zeros);
- `TORCH_SEED=2027` reset every complete experiment;
- no scheduler, dropout, weight decay, or gradient clipping;
- full backpropagation through the MMSE `torch.linalg.solve`;
- CPU execution;
- for the seed, only the final step-2000 weights are judged; no held-out early
  stopping or held-out intermediate checkpoint selection;
- public API `V = model(H,Kq)`;
- final candidate saved as `last_model.pt`.

## 13. Autonomous campaign protocol (C4 — frozen)

The autonomous campaign is authorized only after the SHA-256 hashes of
`prepare.py` and `program.md` are frozen and the seed `train.py` state is committed
in Git. The campaign has a hard maximum of **50 launched research experiments**.
Claude may stop earlier if it concludes that further experiments are unlikely to
produce meaningful progress; before doing so it must append the reason, the main
directions already tested, the stopping experiment number, and the current global
champion score to the research changelog.

### 13.1 Experiment cycle

Every launched experiment follows this exact cycle:

1. Read this charter and the accumulated changelog / `log.csv`.
2. State **one main scientific hypothesis** and one **specific, measurable
   pre-registered falsifier** before editing code. Multiple code changes are allowed
   only when they are necessary parts of that single hypothesis.
3. Verify the director-recorded SHA-256 hashes of `prepare.py` and `program.md`. A
   mismatch aborts before launch and must be corrected before research continues.
4. Edit **only `train.py`** as the mutable research source file.
5. Train the candidate within the allowed training budget in §13.6.
6. Run exactly **one authoritative selection-set evaluation**, only after training is
   complete. No held-out evaluation may be used during training, checkpoint
   selection, or hyperparameter adaptation inside the same experiment.
7. Record the result in both the append-only `train.py` changelog and `log.csv`.
8. Commit/bank or revert according to §13.3 (or the family-local rules in §13.5).

Any new idea motivated by the observed selection result requires a new experiment
ID and consumes a new experiment from the budget.

### 13.2 Research authority

Within `train.py`, Claude may redesign:

- architecture family;
- input representation;
- output parameterization;
- training loss/objective;
- training `Kq` task-sampling law;
- optimizer and learning-rate schedule, batch size, training precision,
  regularization/clipping, and other training-loop details consistent with this charter.

The Baby-v1.1 physics, immutable evaluator, evaluation `Kq=1,...,8`, score
definition, feasibility constraints, selection/final sets, inference contract, and
all other rules outside `train.py` are not research variables.

### 13.3 Global champion and banking

The frozen seed `train.py` is the initial global champion. Its default-seed
(`2027,789`) HELDOUTSCORE is `0.998952`. Before experiment 1, record this seed
incumbent as `run_id=0` in `log.csv` after committing the seed state in Git. A
candidate replaces the global champion only if

`S_new > S_champion + delta`, with `delta = 0.000289`.

Otherwise the candidate is reverted. NaN/Inf, crashes after launch, invalid output,
power infeasibility, inference-budget violation, training-time violation, or any
other evaluator `FAIL / no score` is a revert. `last_model.pt` is a runtime artifact,
not authoritative campaign history; Git plus the append-only records determine the
champion. Older commits and changelog entries must never be rewritten.

For auditability, use this minimal Git/changelog procedure:

1. Commit the candidate `train.py` before launch; that pre-launch candidate commit
   is the `git_commit` recorded for the experiment.
2. On `COMMIT`, append the outcome to the changelog and commit the banked result.
3. On `REVERT` or `FAIL`, restore only the computational code above the append-only
   changelog block; keep the entire current changelog intact, append the new experiment
   entry, and commit that documentation record so no earlier entry is ever lost.
4. During a new-family grace window, keep family-local work on a `family/<label>`
   branch while `main` retains the global champion. Before a family branch is promoted
   or abandoned, append all of that family's changelog entries and `log.csv` rows to
   `main`; `log.csv` is tracked on `main` only. Then promote the best family candidate
   if it wins, or return to the unchanged global champion if it does not.

### 13.4 Architecture-family breadth

At most **6 genuinely different architecture families** may be opened. Every run
must print a stable short family label and the authoritative score, e.g.

```text
FAMILY centralized_mlp
HELDOUTSCORE 1.012345
```

A new family means a materially different computational architecture. Changes to
learning rate, width, features, loss, task sampling, or output parameterization do
not by themselves create a new family unless the computational structure has
materially changed. Claude may open a new family only after stating a concrete
structural limitation of the current approach and a hypothesis for why the new
family may address it. Renaming substantially the same architecture does not evade
the six-family cap.

### 13.5 Five-experiment family grace window

Each newly opened architecture family receives a protected grace window of at most
**5 launched experiments**. During that window the global champion remains
untouched and the family maintains a temporary **family-local champion**.

- Experiment 1 creates the initial candidate for that family.
- Later family experiments start from the best family-local candidate so far.
- The family-local champion is updated only when
  `S_new > S_family + delta`.
- A non-improving family experiment is logged and reverted to the family-local
  champion, not to the global champion.
- After the fifth launched family experiment, the best family-local candidate
  becomes the global champion only if it satisfies
  `S_family > S_global + delta`; otherwise the family is abandoned and research
  returns to the global champion.

A launched `FAIL` consumes both one of the 50 total experiments and, when inside a
new-family grace window, one of that family's five attempts. Pre-launch integrity
or setup failures consume neither budget. Once all six family slots are used,
remaining experiments must perform depth improvements within already opened
families.

### 13.6 Training budget and randomness

The seed/default training length is 2000 optimizer updates. Claude may vary the
training length only as part of the declared hypothesis, within

`500 <= N_steps <= 4000`.

Every launched experiment also has a maximum **10-minute training wall-clock** on
the designated campaign CPU. `train.py` must measure the optimizer-step training
wall-clock and print `TRAIN_S <seconds>`; exceeding 600 s is `FAIL / no score`. The
logged `train_s` is authoritative: `train_s > 600` is `FAIL / no score` regardless of
any editable in-code time-limit value. The chosen `N_steps` and measured `train_s`
must be logged.

Every campaign experiment uses the fixed default seeds
`TORCH_SEED=2027` and `SEED_TRAIN=789`; the calibration-only environment overrides
`CALIB_TORCH_SEED` and `CALIB_TRAIN_SEED` must remain unset during the campaign.
Every run must print the actual seeds used as `SEEDS torch=<...> train=<...>`.
Claude may change the training task-sampling law, but may not search over or change
these underlying campaign seeds.

### 13.7 Falsifiers and interpretation

The falsifier must identify an observable result that would contradict the stated
hypothesis. It may use `HELDOUTSCORE`, one or more per-`Kq` scores, training
stability, feasibility/inference behavior, or another explicitly measured diagnostic
produced by that experiment. The falsifier is for scientific interpretation only;
**only the authoritative HELDOUTSCORE and `delta` rule control banking**.

### 13.8 Required records

Every launched experiment receives a unique run ID that never resets or gets
reused, including `REVERT` and `FAIL` runs. The authoritative narrative record is
append-only inside `train.py`; `log.csv` is the machine-readable record. At minimum,
each `log.csv` row records:

`run_id, family, git_commit, torch_seed, train_seed, torch_version, threads,`
`N_steps, train_s, HELDOUTSCORE, S_KQ1, S_KQ2, S_KQ3, S_KQ4, S_KQ5, S_KQ6,`
`S_KQ7, S_KQ8, inference_s, status`.

`status` is one of `COMMIT`, `REVERT`, or `FAIL`. The narrative changelog must also
record the hypothesis, falsifier, change made, falsifier outcome, result, and short
interpretation. Reverted and failed experiments remain permanently documented.

### 13.9 Campaign termination and final evaluation

When Claude either reaches experiment 50 or invokes the approved early-stop rule,
the best banked **global champion** is the campaign winner. Unfinished family-local
branches are discarded unless they have already legitimately replaced the global
champion. There is **no post-campaign tuning run**.

Claude has zero access to the final set during the campaign. After the global
champion is frozen and all research/tuning has ended, the director exposes the
protected final seed/set and evaluates that champion **exactly once** with the same
immutable scoring procedure. No model, training, hyperparameter, or code change is
permitted after the final result is revealed.
