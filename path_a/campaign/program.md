# Direct 7-cell Percentile Beamforming Autoresearch — Frozen Research Charter

This package is **Path A: Direct 7-cell Autoresearch**. The scientific target,
evaluator, scoring protocol, budgets, and Autoresearch rules are frozen before
experiment 1. During the campaign **only `train.py` may be edited**.

## 1. Frozen physical problem

- 7 hexagonal cells: central cell plus six surrounding cells.
- Hexagon apothem `R = 1000 m`; adjacent-BS spacing `2R = 2000 m`.
- Toroidal wraparound uses shifts `{0, ±t1, ±t2, ±(t1-t2)}`, each nonzero shift
  having magnitude `2R*sqrt(7)`.
- Wraparound image choice is **cell-level nearest-image selection**: for each
  serving-cell/logical-BS pair, the nearest periodic image is chosen from the
  cell-center distances and reused for every user in that serving cell.
- 5 independently and uniformly dropped active users per cell; no scheduling.
- 35 simultaneous users in total.
- BS antennas `M=8`; user receive antennas `N=2`; one stream per user.
- Channel:
  `H = (1 + d/0.392)^(-3.76/2) G`, with i.i.d. `G_mn ~ CN(0,1)`.
- No shadowing.
- Bandwidth `20 MHz`; noise PSD `-143 dBm/Hz`.
- Per-BS total transmit-power constraint `sum_k ||v_k,b||^2 <= 43 dBm`.
  There is no per-user peak-power constraint.
- Perfect instantaneous global CSI is available to one centralized controller.
- Analytical MMSE receive combining; only transmit beamformers are learned.
- Full frequency reuse; all 35 streams transmit simultaneously.
- Serving-BS-only data transmission; no joint transmission, CoMP, SIC, or
  inter-cell data sharing.
- Short-term, single-realization beamforming.
- Rate: `r = 20 log2(1+SINR)` Mbps.

`prepare7.py` is the authoritative implementation of all physics above.

## 2. Frozen objective and task family

For each channel realization, sort the 35 user rates
`r_(1) <= ... <= r_(35)` and optimize the exact network-wide hard SLqP

`F_Kq(V;H) = sum_{i=1}^{Kq} r_(i)(V;H)`.

The authoritative task variable is integer

`Kq in {1,...,35}`.

Percentages `100*Kq/35` are reporting labels only. One shared policy/model must
serve all 35 Kq values; separate per-Kq models are prohibited.

## 3. Immutable evaluator authority

`prepare7.py` is immutable and SHA-256 pinned before the campaign. It owns:

- channel generation and wraparound geometry;
- exact complex128/float64 MMSE/SINR/rate/SLqP scoring;
- feasibility and output-shape checks;
- pinned selection and protected final sets;
- normalization baseline;
- HELDOUTSCORE;
- inference timing and model-state-mutation checks;
- offline MQFT reference.

The evaluator never repairs or projects an invalid candidate output. Invalid
shape/type, NaN/Inf at any rate/SLqP/score stage, per-BS power violation,
inference-budget violation, or any other evaluator contract failure is
`FAIL / no score`. An exactly zero desired transmit beam is valid and must score
that user's rate as exactly zero rather than producing a post-combiner `0/0`.

## 4. Selection and final sets

### Selection

- `N_SELECT = 250` channel realizations.
- `SEED_SELECT = 123`.
- The **same 250 channels** are reused for every `Kq=1,...,35`.
- The selection set controls all commit/revert decisions.
- Selection channels must never enter training or fine-tuning.

### Final

- `N_FINAL = 250`, disjoint from selection and independently seeded.
- The final seed is held outside the repository and supplied only through
  `SEVEN_CELL_FINAL_SEED` after the champion is frozen. It must differ from both
  `SEED_SELECT=123` and `SEED_TRAIN=789`; `prepare7.py` enforces this.
- The same final 250 channels are reused for every Kq.
- The frozen champion is evaluated on final **exactly once**. The director also
  supplies `SEVEN_CELL_FINAL_LOCK_PATH`, pointing to a location outside the
  campaign directory. `evaluate_final` atomically creates that lock with
  exclusive creation before revealing any final score; any second attempt, even
  from a fresh Python process, fails. A failed/aborted claimed final run still
  consumes the one-shot authorization and requires explicit incident reporting.
- No retraining, fine-tuning, hyperparameter changes, checkpoint changes, or
  model/code changes are permitted after final results are revealed.

## 5. Frozen normalization baseline and score

The baseline is **equal-power dominant-direct-channel SVD beamforming**:
for each of the five streams at BS b,

`v_base = sqrt(Pmax/5) * q_max(H_direct)`

where `q_max` is the unit-norm dominant right singular vector of the direct
`2 x 8` user channel.

For each Kq,

`S_Kq = mean_H F_model,Kq(H) / mean_H F_baseline,Kq(H)`.

The authoritative scalar is

`HELDOUTSCORE = (1/35) * sum_{Kq=1}^{35} S_Kq`.

This is **ratio of means**, not mean of per-realization ratios. Every Kq has
equal weight. The frozen baseline has `S_Kq = 1` by construction.

## 6. Offline MQFT reference

MQFT is an **offline comparison reference only** and may never be called by
learned-model inference.

Frozen solver settings:

- initialization: the equal-power direct-SVD baseline in §5;
- relative objective tolerance: `1e-6`;
- maximum iterations: `25`;
- reaching the cap before tolerance is explicitly reported as not converged;
- CVXPY solver order: CLARABEL, then SCS fallback;
- objective decreases and convergence/cap hits are reported.

Because MQFT is expensive, it is reported only at

`Kq in {1,2,4,9,18,35}`

on the same 250 pinned selection channels. Final-set MQFT is protected by the
same final authorization rule.

## 7. Randomness

Three streams are strictly separated:

- training channel stream: `SEED_TRAIN = 789`;
- model/task RNG seed: `TORCH_SEED = 2027`;
- selection: `SEED_SELECT = 123`;
- final: hidden director-held seed, unavailable during the campaign.

Every complete campaign experiment restarts from the same training/model seeds.
Seed hunting or changing the campaign seeds is prohibited. The agent may change
the **training Kq task-sampling law**, but not the underlying fixed RNG seeds.

## 8. Inference contract

At evaluation, trained parameters are frozen. The only permitted operation is a
fixed-cost mapping

`(H, Kq) -> V`.

Forbidden at inference:

- gradient updates or fine-tuning;
- optimizing V separately on held-out instances;
- restarts/candidate searches selected by measured utility;
- adaptive loops such as "iterate until convergence";
- model-state mutation.

Deterministic algebraic post-processing and a fixed predeclared number of
iterations are allowed if they fit the time contract.

The frozen budget is

`T_max = 25.0 s`

for all `250 x 35 = 8750` selection tasks on the designated campaign CPU.
Timing includes only `(H,Kq)->V` inference and excludes channel generation,
scoring, file I/O, and process startup.

Designated campaign hardware follows the Baby campaign convention:
Intel Core i7-14700 CPU with **20 PyTorch intra-op threads** and
**PyTorch 2.14.1+cpu**. Path A and Path B must use the identical machine/software
environment; each run records the exact PyTorch version and thread count.
`requirements.txt` pins this authoritative PyTorch version. Exact NumPy/CVXPY/
CLARABEL/SCS versions were not recorded for the Baby campaign, so they are not
invented here; before experiment 1 the director must record the exact installed
versions alongside the external immutable-hash record and keep them fixed for
both paths.

## 9. Training budget

The starting seed uses exactly `2000` optimizer updates. During Autoresearch,
training length may change only as part of the declared hypothesis and must obey

`500 <= N_steps <= 4000`.

Every launched experiment has a hard optimizer-step training wall-clock cap

`TRAIN_S <= 600 s`.

`train.py` must print `TRAIN_S <seconds>`. A measured value above 600 s is
`FAIL / no score`, regardless of any editable in-code timeout constant.
Evaluation time is not part of `TRAIN_S`.

## 10. Commit/revert threshold

The frozen noise band is

`delta = 0.0005`.

A candidate replaces the applicable champion only if

`S_new > S_champion + 0.0005`.

Operationally the required improvement is `+0.0005`; "±0.0005" is the noise-band
description. FAIL/NaN/Inf/infeasible/time-limit violations automatically revert.

## 11. Source-file authority

Core research/infrastructure files are:

- `prepare7.py` — immutable evaluator;
- `train.py` — **ONLY mutable research source file**;
- `program.md` — immutable research charter;
- `director_rescore.py` — immutable process-separated inference/scoring director;
- `requirements.txt` — immutable campaign environment declaration.

No new source file, notebook, or alternate script may become a hidden location
for research logic. Runtime products such as `log.csv`, checkpoints, diagnostics,
and regenerable caches are allowed.

The narrative experiment changelog is append-only at the end of `train.py`.
Prior entries must never be rewritten or deleted.

`MANIFEST.sha256` inside the package is a convenience copy only and is **not**
the trust anchor. Before experiment 1 the director must keep an authoritative
copy of the immutable SHA-256 values **outside the campaign directory**.
`director_rescore.py` requires this external hash file. The authoritative parent
reads it exactly once at startup, verifies `prepare7.py`, `program.md`,
`director_rescore.py`, and `requirements.txt`, and keeps the trusted digests in
memory for the remainder of that evaluation.

### 11.1 Mutable-train access boundary

The mutable `train.py` may import from `prepare7.py` **only** the frozen public
physical constants listed by `prepare7.audit_train_source` and
`TrainingChannelStream`. `TrainingChannelStream` accepts only the frozen training
seed `SEED_TRAIN = 789`; attempting to instantiate it with any other seed is a FAIL.
It must never import or access the `prepare7` module
object, `_SELECT_H`, `_BASELINE_SELECT_MEAN`, `sample_channels`,
`evaluate_selection`, `evaluate_final`, `run_mqft_reference`,
`mqft_solve_one`, or any private evaluator name. Dynamic evaluator access through
`importlib`, `__import__`, `sys.modules`, function `__globals__`, or direct access
to the training stream's private `_rng` is forbidden. The literal selection seed
`123` is also forbidden in `train.py`. `prepare7.audit_train_source` checks common
direct/dynamic evaluator-access routes
before authoritative evaluation. It is a protocol guard, not a hostile-code sandbox.
The authoritative isolation comes from process separation below.

Before any third-party import, `director_rescore.py` also enforces a package
code allow-list: the campaign directory may contain only `prepare7.py`, `train.py`,
and `director_rescore.py` as `.py` source files. Normal `__pycache__` directories
are ignored because authoritative processes use fresh external cache directories;
other package-local importable artifacts (including `.pyc`, `.pyw`, extension
modules, `.pth`, and `.zip`) and linked/junctioned files or directories are a FAIL.
The authoritative parent re-launches itself under Python isolated mode
(`-I`) when necessary; both workers are also launched with `-I` and fresh
`pycache_prefix` directories.

Mutable `train.py` never evaluates the selection/final set itself. Authoritative
evaluation is split into two fresh processes:

1. **Process A (inference):** imports `train.py`, loads the checkpoint, and computes
   all `V(H,Kq)` values. `load_candidate()` has a separate hard acceptance limit of
   **30 s**. Model output must be an exact `torch.Tensor` (not a subclass), and its
   CPU/complex128 materialization is included inside the frozen 25-s inference
   timing. Loading, artifact file I/O, scoring, and process startup are excluded.
   Process A writes only the precomputed beamformers and trusted timing metadata to
   a temporary artifact. After Process A exits, the parent re-checks the immutable
   files against its original in-memory trusted digests and re-runs the package
   allow-list before Process B may start. Rewriting the external hash file during
   Process A therefore cannot change the accepted hashes.
2. **Process B (scoring):** never imports `train.py`. The parent passes the original
   trusted digests directly to it; Process B does not re-read the external hash file.
   It independently regenerates
   the authoritative channel set, validates the precomputed beams, and computes the
   authoritative SLqP scores and HELDOUTSCORE with immutable `prepare7.py`.

The temporary inference artifact is deleted by the director after scoring. Worker
stdout/stderr is preserved for FAIL diagnosis; Process-A lines are director-prefixed
so they cannot masquerade as an authoritative score. The internal scoring worker
emits only a worker-labelled score; only the parent director
process emits `DIRECTOR_HELDOUTSCORE`, and the commit/revert decision uses only that
parent-director score. Internal worker invocation is not an authoritative campaign
result.

## 12. Path-A initial seed

The direct campaign starts from the generic `train.py` shipped in this package.
It receives **no Baby champion weights, Baby experiment log, or Baby-discovered
architecture/features**.

The fixed SVD baseline is allowed because it is part of the common frozen
7-cell evaluator, not transferred Baby research knowledge.

The seed is intentionally plain:

- centralized full-CSI residual MLP;
- input = scaled full CSI, frozen SVD-baseline directions, and `Kq/35`;
- output = baseline plus learned complex correction, then per-BS feasibility
  scaling only when necessary;
- exact raw hard-SLqP training loss;
- Adam with `lr=1e-3`;
- `2000` updates, batch `16`, fresh channels each update;
- independent `Kq ~ Uniform{1,...,35}` per training sample;
- float32 neural network and complex128/float64 differentiable MMSE physics;
- no scheduler, dropout, weight decay, or gradient clipping;
- only the final checkpoint is evaluated, through `director_rescore.py`'s process-separated inference/scoring pipeline.

These seed details are starting conditions, not frozen research variables: the
agent may redesign them inside `train.py` subject to this charter.

## 13. Autonomous experiment protocol

The campaign has a hard maximum of **50 launched experiments**.

Every launched experiment must:

1. Read this charter and accumulated changelog/log.
2. State **one primary scientific hypothesis** and **one explicit measurable
   pre-registered falsifier** before editing code.
3. Verify the director-held external immutable hashes before launch.
4. Edit only `train.py`. `prepare7.audit_train_source` must pass.
5. Train within §9 and save the final candidate checkpoint as `last_model.pt`.
   `train.py` may not access the selection set or call evaluator scoring.
6. Run immutable `director_rescore.py` with the director-held external hash
   file. It launches the frozen **Process A inference -> Process B scoring**
   pipeline in separate fresh processes. This is the experiment's exactly one
   authoritative selection evaluation. No selection evaluation may be used for
   within-run early stopping, checkpoint selection, or hyperparameter adaptation.
7. Append the director score/result to the permanent changelog and `log.csv`.
8. Commit/bank or revert according to the frozen `delta=0.0005` rule using the
   director score only.

Multiple code changes in one experiment are allowed only when they are necessary
parts of the single declared hypothesis.

A launched `COMMIT`, `REVERT`, or `FAIL` consumes one of the 50 experiments.
Pre-launch integrity/setup failures consume no experiment.

The agent may stop before 50 only after recording why further search is unlikely
to help, the major directions already tested, the stopping run ID, and the
current global champion score.

## 14. Research authority inside train.py

The agent may redesign:

- architecture family;
- input representation/features;
- Kq representation;
- transmit-beamformer output parameterization;
- training loss/objective;
- training Kq task-sampling law;
- optimizer and learning-rate schedule;
- batch size and training precision;
- regularization/clipping and other training-loop details;

provided all immutable evaluator, feasibility, timing, seed, selection/final,
and campaign rules remain satisfied.

Regardless of architecture changes, `train.py` must retain a public
`load_candidate(checkpoint_path)` function that reconstructs the saved candidate
without training or held-out access. This is the only mutable-side interface used
by `director_rescore.py`.

## 15. Architecture-family breadth and grace window

At most **6 genuinely different architecture families** may be opened.

A new family must represent a materially different computational architecture,
not merely a renamed width/lr/feature tweak. Before opening one, the agent must
state a structural limitation of the current approach and why the new family
could address it.

Each newly opened family receives a protected grace window of at most
**5 launched experiments**:

- the global champion remains untouched during the window;
- experiment 1 creates the initial family candidate;
- later family experiments start from the best family-local candidate;
- family-local champion updates require `S_new > S_family + 0.0005`;
- after at most five launched attempts, the family replaces the global champion
  only if `S_family > S_global + 0.0005`; otherwise the family is abandoned.

A launched FAIL consumes both one total experiment and one family-grace attempt.
Once all six family slots are used, remaining experiments must deepen already
opened families.

## 16. Required records

Each launched experiment receives a unique run ID that never resets or gets
reused. At minimum `log.csv` records:

`run_id, family, git_commit, torch_seed, train_seed, torch_version, threads,`
`N_steps, batch, train_s, HELDOUTSCORE, inference_s, status,`
`S_KQ1, ..., S_KQ35`.

`HELDOUTSCORE` and `inference_s` are the values printed by the fresh-process
`director_rescore.py`; self-reported substitutes from mutable code are not valid.
`status` is `COMMIT`, `REVERT`, or `FAIL`.

The append-only changelog in `train.py` additionally records the hypothesis,
falsifier, code change, falsifier outcome, authoritative result, and short
interpretation. Reverted and failed experiments remain permanently documented.

## 17. Campaign termination and final evaluation

At experiment 50, or an approved documented early stop, the best banked global
champion is frozen. There is **no post-campaign tuning run**.

Only after that freeze may the director expose `SEVEN_CELL_FINAL_SEED`, choose an
external unused `SEVEN_CELL_FINAL_LOCK_PATH`, and run the one-shot final
evaluation. The lock is atomically consumed by `prepare7.evaluate_final`; a
second attempt fails. No model or code changes are permitted after the final
result is revealed.

## 18. Path isolation

Path A and Path B use the same 7-cell target, evaluator, score, selection/final
sets, seeds, budgets, inference contract, and Autoresearch rules.

Their only intended difference is starting research state:

- **Path A (this package):** generic direct 7-cell seed; no Baby champion/log/
  discoveries.
- **Path B:** mechanical 7-cell adaptation of the frozen Baby champion/
  discoveries, performed without 7-cell score-guided human tuning.

After either campaign starts, **no information may flow between Path A and
Path B**.
