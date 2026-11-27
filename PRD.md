# Product Requirements Document
## Multiverse of Monsoon — Physics-Aware Regional Atmospheric Digital Twin

## 1. Document metadata and version history

| Field | Value |
|---|---|
| Version | 1.0 — architectural review draft |
| Date | 2026-11-26 |
| Status | **READY FOR ARCHITECTURAL REVIEW — standalone team edition** |
| Scope | Planning, research, architecture, testing, and compliance only |
| Implementation authorization | **Not granted** |
| Official kit | `v1.0.0`, commit `689c8798bbff7f318a1e74878a6985abc646080a` |
| Primary authorship mode | Team-directed engineering plan; no experimental results claimed |

Change control: material changes to channel order, units, perturbation semantics, split, primary/fallback architecture, evaluation definitions, or compliance interpretation require an ADR and team approval.

This file is the single shareable engineering blueprint. It contains the requirements, evidence, architecture, experiments, tests, roadmap, risks, compliance gates, and decisions needed by the team. Separate Markdown files in the workspace are evidence ledgers and working views, not prerequisites for understanding this PRD.

## 2. Executive summary

Build a CPU-efficient regional atmospheric state-transition system that maps three daily Indian-domain states to up to 14 future daily states and uses the same learned transition after an optional one-time physical-unit perturbation. The official score rewards forecast skill, heavy rain, retained spatial variance, and localized stable shock propagation under a 5-second ten-step limit.

The measured dataset is small-grid but substantial in time: 6,209 continuous days, six dynamic fields on 32×36 cells, plus fractional land-sea mask and orography. It has high one-day persistence, strong annual/geographical structure, and unusually difficult precipitation tails. It lacks vertical profiles, lateral boundary forcing, radiation/cloud/soil/ocean state, and counterfactual labels. The product is therefore an empirical regional atmospheric digital twin, not a complete numerical weather model.

Proposed strategy:

1. Establish persistence, permitted seasonal climatology, and regularized linear baselines.
2. Build a compact shared-weight residual CNN with nonperiodic multiscale spatial processing, static/coordinate conditioning, and supplied seasonal phase.
3. Train progressively on autoregressive rollouts, starting with a simple normalized robust loss.
4. Introduce at most one extreme/sharpness loss at a time after measured diagnosis.
5. Test metric-aware proxy transport only as an optional ablation; abandon it if it does not improve skill/causality within compute budget.
6. Maintain an independently trained simpler residual CNN fallback.

Primary candidate, fallback, novelty hypotheses, resource targets, and split all require approval. Nothing here is experimentally validated.

## 3. Project vision

Create a scientifically honest prototype that forecasts plausible regional atmospheric evolution and lets reviewers inspect how unfamiliar localized changes affect the same learned dynamics. Optimize for reproducible evidence, generalization, stability, and compliance—not architectural fashion or scripted demonstrations.

## 4. Exact problem statement

**VERIFIED OFFICIAL REQUIREMENT.** Given `init_states[3,6,32,36]`, supplied last-day `doy_sincos`, `steps`, and a nullable physical delta `perturbation[6,32,36]`, return `rollout[steps,6,32,36]` in physical units. Timeline Prime uses the unmodified latest state. Multiverse adds the delta once to the latest state and uses the same engine. The scored path is ten days; contract code supports at most 14. See problem statement pages 7–10 and official contract lines 5–53.

## 5. Stakeholders and intended users

- Automated evaluator: contract-valid, fast, deterministic outputs.
- Hackathon technical judges: scientific reasoning, original work, live new-shock response, limitations.
- Team researchers/engineers: reproducible experiments and modular interfaces.
- Demonstration users: compare base and perturbed trajectories and qualified impact proxies.
- Future researchers: extend data/state/boundaries without rewriting the core evaluation framework.

## 6. Goals and non-goals

### Goals

- Beat persistence at day 4 and improve over the stronger persistence/climatology reference.
- Stable finite 10–14-day autoregression with nonnegative precipitation and retained structure.
- Continuous, bounded, localized response to arbitrary valid perturbations; exact zero consistency.
- Strong heavy-rain and tail diagnostics without indiscriminate rain inflation.
- CPU-only offline inference with margin under official limits.
- Live dashboard/impact output derived from submitted engine.
- Complete provenance, tests, fallback, and honest claim boundaries.

### Non-goals

- Full primitive-equation/Navier–Stokes or coupled climate simulation.
- Operational warnings or official heatwave declarations.
- Counterfactual causal truth without intervention data.
- External forecast APIs/data/checkpoints at inference.
- Global model reproduction, distributed microservices, LLM/chatbot features, or scenario scripts.

## 7. Source inventory

The consolidated inventory and hashes are included in Appendix F. The only unresolved organizer artifact is the Canva presentation export. PDF, release, code, contract, validator, scoring, scanner, dataset, statistics, and license were inspected.

## 8. Verified organizer constraints

The consolidated mandatory-requirement matrix is included in Appendix G. Official limits: 4 CPU cores, 6 GB RAM, no GPU, offline/read-only repository, `/tmp` writes only, readiness ≤120 s, ten steps <5 s wall time, response <80 MB, root `serve.py`, weights local, repository/weights under stated 2 GB, and original work. Canonical wire format is finite nested JSON lists in physical units.

## 9. Requirements traceability

The traceability matrix assigns `FR`, `API`, `DATA`, `NFR`, `SCI`, `EVAL`, and `COMP` identifiers, sources, mandatory status, acceptance tests, and consequences. No mandatory item is accepted solely by documentation; implementation evidence must populate the matrix later.

## 10. Dataset analysis

The complete decision-relevant audit is consolidated in this section and Appendix H. Critical observations:

- Shape `[6209,6,32,36]`, 2005-01-01–2021-12-31, no time gaps/duplicates/NaN/Inf.
- Ascending 1° coordinates; longitudinal physical width shrinks about 110.5→88.2 km south→north.
- Lag-1 correlation: temperature .997, precipitation .712, winds .936/.907, pressure .947, water vapour .985.
- Precipitation median .551, P99 35.70, max 249.77 mm/day; 7.81% exact zeros.
- Fractional coastline and terrain from approximately sea level to 5,284 m.
- Strong seasonal and land/ocean contrasts; supplied mean/std reproduce full-data statistics.

Architectural consequence: local spatial processing and residual forecasting are plausible; precipitation needs separate evaluation; metric/edge/geography information matters; random splits would leak highly correlated windows.

## 11. Scientific background

Weather state evolution may be represented abstractly as `X_{t+1}=F(X_{t-h+1:t}, G, S_t)+ε`, with dynamic history `X`, geography `G`, seasonal phase `S`, unresolved forcing, and model error `ε`. A data-driven transition approximates conditional evolution in the training distribution. Autoregression repeatedly evaluates the transition on its own predictions, causing distribution shift and compounding error.

Hybrid weather literature shows value in tendency prediction, multi-step training, and explicit dynamics, but global systems use vertical state and forcing absent here. Compatibility judgments are included in sections 13–14.

## 12. Atmospheric modelling assumptions

| Statement | Category | Boundary |
|---|---|---|
| Daily fields contain predictive local/spatial structure | Experimental hypothesis supported by correlations | Must be validated chronologically |
| `u10/v10` provide directional proxy information | Physically motivated approximation | Not column wind |
| `tcwv` can be transported approximately | Approximation | Sources/sinks/boundary exchange prevent conservation |
| Coordinates/terrain/mask condition local dynamics | Engineering hypothesis | Association does not prove causation |
| Supplied seasonal phase aids generalization | Contract-permitted hypothesis | No dates/month lookup |
| Region has open boundaries | Established limitation | No exact closed-domain balance |

## 13. Research literature review

- GraphCast: multiscale graph global model; relational ideas only.
- NeuralGCM: differentiable core plus learned physics; tendency and rollout curriculum inspiration only.
- Pangu-Weather: global 3D transformer and hierarchical lead models; direct adoption unsuitable.
- FourCastNet/FNO: efficient long-range spectral processing; periodic/global bias risky here.
- ConvLSTM: local recurrent sequence model; viable challenger but potentially slower and harder to stabilize.
- WeatherBench 2: lead-wise, latitude-weighted, baseline-relative evaluation principles.

No external checkpoint or weather dataset is proposed.

## 14. Open-source compatibility matrix

The organizer kit is directly reusable as contract/test tooling; WeatherBench ideas are adaptable; ConvLSTM/FNO methods are adaptable only as experiments; GraphCast/NeuralGCM/FourCastNet/Pangu full systems are research inspiration or unsuitable. All adopted code requires a pinned version, license entry, and attribution.

## 15. Architecture alternatives

| Candidate | Parameters (estimate) | CPU/complexity | Strength | Primary risk |
|---|---:|---|---|---|
| Persistence/climatology/ridge | 0–small | Minimal | Essential references, robust | Cannot model nonlinear shocks well |
| Simple residual CNN | 0.3–2M | Very low | Strong locality, stable fallback | Limited receptive field |
| Geography-aware multiscale residual CNN | 1–5M | Low | Best balance; static/metric context | Padding/boundary and blur |
| Proxy-physics + learned residual | 1–5M + fixed ops | Low–medium | Directional interpretability | Wrong physics from incomplete state |
| ConvLSTM | 1–8M | Medium | Recurrent memory | Sequential latency/training complexity |
| Small FNO/multiscale operator | 1–8M | Low–medium | Long-range coupling | Artificial periodicity/small grid |
| Transformer/GNN/global models | large | High | Expressivity | Unjustified scale/data/compute |

## 16. Architecture decision matrix

The weighted matrix below gives the proposed geo-residual CNN 425/500, residual CNN 410, physics-guided 395, ConvLSTM 335, small FNO 325, and baselines 300. These are design estimates, not measurements.

## 17. Recommended architecture

### Minimum viable system

Persistence/climatology/ridge evaluation; compact residual CNN; shared autoregressive path; internal normalization; exact API; official validator/scanner; independently trained fallback; minimal live comparison dashboard.

### Primary candidate: GRAST (working name)

**Geography-aware Residual Atmospheric State Transition**:

- Input: three normalized dynamic states (18 channels), continuous LSM, normalized orography, latitude, longitude, `cos(lat)`/metric features, and broadcast supplied `doy_sincos`.
- Encoder: shallow nonperiodic convolutions.
- Core: compact residual blocks at native and one coarser scale; skip fusion preserves fine structure.
- Output: normalized per-channel one-day tendencies; precipitation head/transform remains an experiment.
- Rollout: same transition and weights each day; seasonal pair advanced analytically.
- No shock-specific inputs beyond the perturbed state.

### Fallback

A smaller single-scale residual CNN using dynamics + static fields and the same rollout/API. It must be trained and validated, not generated after failure.

### Optional extensions

Metric-aware proxy transport, one rain-tail loss, one gradient loss, lightweight ensemble/OOD indicator—each gated by [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

## 18. Complete system architecture

```mermaid
flowchart LR
  subgraph O[Organizer-provided]
    NC[NetCDF + stats]
    CT[Contract + validator + scanner]
  end
  subgraph T[Training only]
    DV[Data validation] --> W[Chronological windows]
    W --> B[Baselines]
    W --> M[Candidate training]
    M --> EV[Scientific evaluation]
    B --> EV
    EV --> EX[Selected primary + fallback export]
  end
  subgraph I[Offline inference]
    API[serve.py /simulate] --> PP[Validate + normalize + perturb once]
    PP --> RE[Shared rollout engine]
    EX --> RE
    RE --> PO[Denormalize + validity + JSON]
  end
  subgraph U[Separate user experience]
    UI[Interactive map] --> API
    API --> CMP[Prime vs Multiverse comparison]
    CMP --> HR[Heat-risk proxy + diagnostics]
  end
  NC --> DV
  CT --> API
```

## 19. Mathematical formulation

Let `X_t∈R^{6×32×36}`, static `G`, seasonal pair `s_t`, normalization `N_c(x)=(x-μ_c)/σ_c`. Perturb once:

`X'_t = X_t + δ`, with `δ=0` for Timeline Prime/null.

History `H_t=[N(X_{t-2}),N(X_{t-1}),N(X'_t),G,s_t]`. Learned tendency:

`ΔZ_t = f_θ(H_t, P_t)` and `Z_{t+1}=N(X'_t)+r⊙ΔZ_t`, then `X_{t+1}=N^{-1}(Z_{t+1})`.

`r` is a learned/fixed per-channel residual scale chosen from training data, not a physical time integrator. For later steps, roll history forward and compute `s_{t+k}` analytically from the supplied phase. The same `f_θ` is used in both universes.

Optional proxy transport for scalar-like field `q`:

`A(q) = -(u_p ∂q/∂x + v_p ∂q/∂y) + κ∇²q`,

where derivatives use metric `dx(lat),dy`, nonperiodic boundaries, and proxy winds. This is not exact moisture dynamics. Candidate update becomes `ΔZ=f_θ(H,A)+λA`; `λ/κ` are bounded/trainable or selected on validation. Reject if stability/skill/causality do not improve.

## 20. Detailed module specifications

| Module | Inputs → outputs | Responsibility / errors | Budget / tests | Priority |
|---|---|---|---|---|
| DatasetReader | NetCDF → typed arrays/coords | Verify hash/schema/order/units; fail closed | Training only; audit tests | P0 |
| WindowIndex | dates/split/horizon → indices | No cross-split overlap | Exact temporal sentinels | P0 |
| Transform | physical arrays ↔ normalized | Single source of means/scales; finite checks | <2 ms target; round trip | P0 |
| GeoFeatures | coords/static → fixed channels | LSM/orography/metric features | Bundled small tensors | P1 |
| Baselines | history/DOY → rollout | Persistence/climatology/ridge references | Metric parity | P0 |
| Transition | history/features → tendency | Shared one-day dynamics | ≤5M target; shape/determinism | P0/P1 |
| ProxyOperator | state/metrics → proxy tendencies | Optional finite differences, nonperiodic | <10% latency overhead | P2 |
| RolloutEngine | history/steps/δ → future states | Apply δ once; advance phase; shared recurrence | 1–14 steps; stability tests | P0 |
| Diagnostics | truth/base/shock → metrics | Skill, rain, blur, shock, strata | Exact official parity | P0 |
| ExportLoader | artifact → initialized model | Hash/version/load once | cold-start/RAM | P0 |
| APIAdapter | JSON ↔ arrays | Contract, timing, errors, serialization | official validator | P0 |
| DashboardClient | user controls ↔ API | Live visualization only | provenance/integration | P0 demo |
| HeatRiskProxy | rollout/climatology → zones | Qualified anomaly/persistence proxy | scientific review | P0 demo |

Public interfaces and errors must be frozen before parallel implementation. Malformed requests should return deterministic 4xx without exposing internals; the official starter behavior is the baseline where the contract is silent.

## 21. Custom algorithms and research contributions

### Contribution A — metric- and geography-aware regional transition

- Problem: ordinary image CNNs treat index distances uniformly and ignore open regional boundaries.
- Adaptation: continuous terrain/coast/coordinate channels, latitude-dependent metric features, nonperiodic padding, edge-stratified validation.
- Hypothesis: improve terrain/coast/edge skill and perturbation locality with negligible inference overhead.
- Ablation: dynamic-only → static → coordinate/metric.
- Retain: consistent validation improvement, especially edge/terrain strata, with no compute/gate regression.
- Abandon: no gain across seeds or boundary artifacts increase.

### Contribution B — perturbation-faithful shared rollout and diagnostic suite

- Problem: no counterfactual labels, but official evaluation demands nontrivial local propagation.
- Adaptation: one shared transition plus systematic zero, scale, channel, geometry, direction, influence-radius, stability, and boundary diagnostics.
- System novelty: counterfactual behavior is treated as a first-class test surface, not a demo animation.
- Retain criteria: exact null consistency; organizer shock checks; continuous magnitude response; no hardcoded scenario logic.

### Experimental contribution C — proxy transport plus learned closure

- Problem: purely learned local dynamics may spread shocks diffusively or ignore direction.
- Foundation: advection-diffusion as an approximate inductive bias.
- Modification: metric-aware fixed differentiable proxy supplied as a feature/tendency, with learned residual closure.
- Hypothesis: improve shock propagation and multi-day moisture/wind structure.
- Overhead target: <10% ten-step latency.
- Abandon: no robust gain, numerical sensitivity, teleportation, or false conservation implication.

No claim of global algorithmic novelty is made. A and B are original system-level adaptations; C is a research hypothesis.

## 22. Training methodology

1. Proposed chronological partitions: train 2005–2017, validation 2018–2019, locked test 2020–2021; approve before coding.
2. Construct windows wholly inside one partition for three inputs and up to 14 targets.
3. Use organizer statistics for contract parity; repeat decisive comparisons with train-only normalization to quantify full-data-stat sensitivity.
4. Train deterministic one-step baseline first; then progressive horizons such as 2→4→8/10 with bounded memory.
5. Use mixed teacher/self-fed schedules only when simpler unrolled training is insufficient; record schedule explicitly.
6. Fixed seeds, identical budgets, validation-only selection, three seeds for retained comparisons.
7. Export primary and fallback with config, code commit, data hash, stats hash, and artifact hash.

```mermaid
flowchart LR
  NC[Verified NetCDF] --> VA[Schema/quality audit]
  VA --> SP[Chronological split]
  SP --> WI[Nonoverlapping windows]
  WI --> TF[Training transforms]
  TF --> BL[Baselines]
  TF --> TR[Candidate training]
  TR --> VR[Validation + ablations]
  BL --> VR
  VR --> LT[One-time locked test]
  LT --> PK[Primary/fallback package]
```

## 23. Loss-function design

Start with normalized channel-balanced Huber:

`L_state(k)=Σ_c w_c mean Huber((ŷ_{k,c}-y_{k,c})/σ_c;β)` and `L=Σ_k α_k L_state(k)`.

- `w_c`: begin equal contribution; not inverse-frequency guesswork.
- `α_k`: begin uniform over trained horizon; test modest later-lead emphasis only if needed.
- Huber reduces extreme gradient domination but may weaken extreme learning; compare MSE.

Optional, one at a time:

- Rain-tail: capped target-dependent weight or two-part occurrence/intensity objective.
- Gradient: finite-difference mismatch to address diagnosed blur.
- Stability regularizer: penalize only measured out-of-range normalized tendencies, not training extrema.

Every term needs weight sweep, ablation, overall/tail trade-off, seed consistency, and cost. Do not optimize official variance by injecting noise.

## 24. Rollout strategy

```mermaid
flowchart LR
  H0[X t-2, X t-1, X t] --> F1[Shared transition]
  F1 --> X1[X t+1]
  X1 --> H1[Shift history]
  H1 --> F2[Same weights]
  F2 --> X2[X t+2]
  X2 --> R[Repeat to requested steps ≤14]
```

Residual one-step autoregression is preferred over direct 10-step output because it supports arbitrary permitted horizons and shared perturbation dynamics. Risks are compounding error and exposure bias; progressive unrolling is the primary remedy. Direct multi-step/cascaded lead models are rejected initially because they multiply artifacts/complexity and complicate counterfactual consistency.

## 24.1 Autoregressive error accumulation and long-horizon drift

**Architectural priority:** rollout error is a first-order design constraint, equal in importance to one-day accuracy. A model is not acceptable merely because it predicts day 1 well. Architecture selection, training, checkpoint selection, and release gates must explicitly measure degradation at days 1, 4, 8, and 10 in both Timeline Prime and unfamiliar perturbed trajectories.

### Why error compounds

Let the true normalized transition be `z_{t+1}=F(z_t)+ξ_t` and the learned transition be `ẑ_{t+1}=f_θ(ẑ_t)`. With rollout error `e_t=ẑ_t-z_t`, a first-order expansion gives

`e_{t+1} ≈ J_F(z_t)e_t + b_θ(z_t) - ξ_t`,

where `J_F` is the local transition Jacobian and `b_θ=f_θ-F` is model bias. Repeated substitution yields

`e_{t+k} ≈ (Π_{j=0}^{k-1}J_{t+j})e_t + Σ_{i=0}^{k-1}(Π_{j=i+1}^{k-1}J_{t+j})(b_{t+i}-ξ_{t+i})`.

Therefore even small one-step bias can accumulate, and errors aligned with locally amplifying modes can grow rapidly. In this regional setting, amplification can differ by variable and location: precipitation errors may alter learned moisture coupling; wind errors may displace anomalies; pressure/temperature bias may spread across channels; boundary errors can propagate inward. A shock also moves the model away from the ordinary training-state distribution, so perturbation rollouts require separate degradation analysis.

This Jacobian analysis motivates stability diagnostics but does **not** justify globally forcing the learned transition to be contractive. Real atmospheric disturbances can grow; excessive contraction would create the prohibited smooth climatological collapse.

### Candidate solutions

| Candidate | Mechanism | Expected benefit | Principal risk | Required evidence |
|---|---|---|---|---|
| Multi-step autoregressive training | Unroll the shared model through its own outputs and backpropagate across multiple leads | Directly optimizes the state distribution encountered at inference | Memory/training cost; later-lead objectives can weaken day 1 | Compare 1-step, 2→4→8, and 2→4→8→10 curricula under equal budgets |
| Scheduled sampling | Replace selected ground-truth history states with model predictions using a declared probability schedule | Gradual exposure to self-generated inputs | Biased/inconsistent training objective; schedule sensitivity | Compare no sampling, linear/probability ramp, and full unroll; retain only if seed-robust |
| Residual/tendency prediction | Predict scaled normalized `Δz` rather than the full next state | Leverages high persistence and makes update magnitude observable | Systematic tendency bias accumulates; rain residuals can become negative | Direct-state versus residual target with identical backbone and constraints |
| Lead-time-weighted objective | `L=Σ_{k=1}^H α_k L_k`, with declared normalized weights | Trades short-lead accuracy for longer-horizon robustness deliberately | Arbitrary weights can hide day-1 regression or overfit official leads | Uniform versus mild increasing weights; report every lead separately |
| Physics-aware stability constraints | Penalize implausible tendencies, scale drift, unstable gradients, and optionally use bounded proxy transport | Discourages numerical modes without hiding them after inference | Incorrect constraints suppress real extremes or shocks | Constraint-by-constraint ablation; tail/sharpness/causality safeguards |
| Direct multi-horizon head | Predict days 1…10 jointly from the initial history | Removes recursive feedback inside the prediction call | Fixed horizon, inconsistent intermediate dynamics, larger output/head, weak arbitrary-shock evolution | Same parameter/compute comparison to autoregression; per-lead consistency tests |
| Hybrid/direct anchors | Autoregressive core plus sparse learned corrections or direct anchors at selected leads | May reset drift while retaining stepwise dynamics | Multiple transition semantics and discontinuities | Only after simple autoregression fails; continuity and shared-engine review |

### Proposed training sequence

1. Establish direct-state and normalized-tendency one-step baselines.
2. Select the more stable target using day-1/4/8/10 validation, not day 1 alone.
3. Increase differentiable rollout horizon progressively, for example 2→4→8→10, only after each horizon is numerically stable.
4. Compare full self-fed unrolling with a scheduled-sampling ramp. The schedule must be a logged configuration, not an implicit training heuristic.
5. Start with uniform lead weights. Test a mild normalized increasing schedule only if days 8/10 degrade disproportionately; do not zero or trivialize day-1 loss.
6. Add one stability constraint only after diagnosing a corresponding failure. Never use post-hoc hard clipping to claim stability.
7. Select the checkpoint with a rollout-aware validation rule containing hard validity and separate day-1/4/8/10 terms.

### Physics-aware stability candidates

These are EXPERIMENTAL HYPOTHESES, not mandatory physics:

- **Tendency-scale penalty:** penalize normalized updates beyond a training-derived high quantile with a smooth function, while auditing extreme targets separately.
- **State-scale drift penalty:** compare predicted spatial mean/std/quantiles with target statistics at each lead; use only a weak differentiable penalty if unconstrained drift is observed.
- **Gradient/spectral consistency:** discourage loss of or artificial inflation of spatial structure; never optimize variance alone.
- **Semantic precipitation transform:** test a nonnegative parameterization or occurrence/intensity head against an unconstrained residual plus training penalty. Reject if it erases light rain or heavy-rain tails.
- **Metric-aware proxy stability:** if proxy transport is retained, bound learned coefficients and test nonperiodic boundary/CFL-like sensitivity. This is a numerical safeguard for an approximation, not a proof of physical stability.
- **Jacobian/sensitivity monitoring:** estimate local perturbation amplification with finite differences or Jacobian-vector products during validation. Use as a diagnostic first; regularize only if it predicts observed blow-up without suppressing legitimate response.

### Error-propagation analysis

For each channel `c`, region `r`, lead `k`, and scenario `s`, record normalized error

`E_{c,r,s}(k)=RMSE(ŷ_{k,c,r},y_{k,c,r})/σ_c`

and incremental degradation

`G_{c,r,s}(k)=E_{c,r,s}(k)-E_{c,r,s}(1)`.

Also compute error cross-correlation `Corr(e_{k,c},e_{k,c'})`, regional error-energy migration, boundary-to-interior error, bias, predicted/true scale ratio, gradient/spectral ratio, and heavy-rain errors. Stratify at minimum by land/ocean, terrain bands, boundary/interior, monsoon/non-monsoon season, heavy-rain cases, and perturbation location. For artificial perturbations without counterfactual truth, compare perturbed-vs-base response stability and sensitivity; do not report forecast RMSE as counterfactual accuracy.

### Direct multi-horizon versus autoregressive decision

The shared autoregressive transition remains the preferred candidate because it naturally supports steps 1–14 and repeated shock evolution. A direct multi-horizon model is a mandatory challenger experiment, not the default. Compare:

- identical input/static/seasonal information;
- approximately matched parameters and training compute;
- days 1/4/8/10 skill, extremes, sharpness, perturbation continuity, latency, and artifact size;
- temporal consistency, including whether its day `k+1` prediction agrees when initialized from its day `k` state;
- ability to serve arbitrary permitted step counts.

Retain a direct or hybrid approach only if it provides a clear Pareto improvement and preserves a coherent common engine for normal and perturbed simulations.

### Required ablation matrix

| Ablation | Controlled comparison | Primary readout |
|---|---|---|
| A-DRIFT-1 | Direct state vs normalized tendency | Error growth and bias at 1/4/8/10 |
| A-DRIFT-2 | One-step training vs 2→4→8→10 unroll curriculum | Late-lead skill and stability |
| A-DRIFT-3 | Teacher forcing vs scheduled sampling vs fully self-fed unroll | Exposure-gap reduction across seeds |
| A-DRIFT-4 | Uniform vs mildly increasing `α_k` | Day-8/10 gain versus day-1/extreme trade-off |
| A-DRIFT-5 | No constraint vs each stability constraint individually | Validity/scale gain without smoothing/tail loss |
| A-DRIFT-6 | Autoregressive vs matched direct multi-horizon challenger | Pareto comparison and temporal consistency |
| A-DRIFT-7 | Base states vs unfamiliar perturbation matrix | Shock-conditioned amplification and locality |
| A-DRIFT-8 | Full domain vs edge/terrain/land-ocean strata | Geographic concentration/migration of errors |

### Measurable acceptance criteria

Final numeric targets must be frozen after mandatory baselines are measured; they must not be chosen post hoc to favor a model. Until then, the following criteria are binding:

1. Report independent day-1, day-4, day-8, and day-10 results for every dynamic channel; averages alone are insufficient.
2. Meet the official day-4 persistence hard gate and all validity/anti-blur/causality requirements.
3. Relative to the one-step-trained version of the same backbone, the selected rollout strategy must improve the predeclared aggregate normalized RMSE at both days 8 and 10 in at least two of three seeds, with no material day-1 regression. Define “material” before running the comparison; proposed initial tolerance is 2% relative.
4. No retained method may improve late-lead RMSE by causing any channel’s day-8 spatial-variance ratio to leave the official `[0.40,2.50]` range, or by materially degrading gradient/spectral fidelity.
5. Heavy-rain CSI skill at days 1/4/8 and an internal day-10 CSI must not materially regress; proposed initial tolerance is 0.02 absolute CSI, subject to baseline variance.
6. All zero/null perturbation comparisons must satisfy the official `1e-3`-of-channel-std tolerance. Nonzero unfamiliar perturbations must remain finite, bounded, continuous with magnitude, initially localized, and free of artificial teleportation.
7. No stability claim may depend on hard clipping to training extrema or replacing failed predictions with persistence.
8. Report channel-to-channel and region-to-region error propagation; investigate any late-lead channel/region degradation exceeding twice its day-1 normalized error before release.
9. The selected strategy must remain inside the inference budget; training-time complexity is acceptable only when reproducible within the event schedule.

Failure of these criteria triggers simplification, retraining, or selection of the validated fallback. A model that is sharp but unstable, stable but over-smoothed, or accurate on ordinary states but fragile under unfamiliar valid perturbations is not release-ready.

## 25. Perturbation methodology

```mermaid
flowchart TD
  I[Three-day initial state] --> B{Branch}
  B -->|Prime| P[Latest state unchanged]
  B -->|Multiverse| A[Add physical delta once]
  P --> E[Shared evolution engine]
  A --> E
  E --> RP[Prime rollout]
  E --> RM[Multiverse rollout]
  RP --> D[Difference and diagnostics]
  RM --> D
```

Required diagnostics: normalized maximum response; day-1 near-energy; outside-neighborhood energy over time; energy-weighted centroid and radius; cross-channel response matrix; magnitude-response curve; decay/amplification; boundary/teleport fraction; validity/stability. Exact official functions are reproduced in test code, while broader tests prevent overfitting to named shocks.

Counterfactual plausibility is not causal validation. No historical target can establish what would have happened after an artificial intervention.

## 26. Scientific diagnostics

```mermaid
flowchart LR
  O[Rollouts] --> SK[RMSE/skill by lead/channel]
  O --> RN[Rain CSI/precision/recall/tails]
  O --> SH[Variance/gradients/spectra]
  O --> ST[Scale/drift/validity]
  O --> GS[Land/ocean/terrain/edge strata]
  O --> PT[Shock locality/propagation/stability]
  SK & RN & SH & ST & GS & PT --> DC[Evidence dashboard + decision gate]
```

Metrics must be reported individually; a composite never hides gate failures or rain/extreme regressions.

## 27. Generalization strategy

- Chronological locked period; no random overlapping split.
- Seasonal/event/terrain/land-ocean/edge stratification.
- Inputs from distribution tails and physically unusual cross-channel combinations, distinguished from malformed inputs.
- Arbitrary valid perturbation channel/shape/sign/magnitude/location.
- Determinism, statelessness, repeated/interleaved calls, permitted lengths.
- No dates/years/event labels in inference; only state, geography, and contract pair.
- Report OOD indicators only if calibrated against held-out errors; otherwise report descriptive distance, not uncertainty.

## 28. Evaluation methodology

Official parity:

- Latitude-weighted RMSE skill `1-RMSE_model/min(RMSE_persist,RMSE_clim)` at days 1,4,8.
- Rain CSI at >10 mm/day and skill over better reference.
- Day-8 spatial variance ratio per channel.
- Public causality checks and hard validity gate.
- Unmodified validator and scanner.

Broader internal metrics: MAE, bias, RMSE by channel/lead; rain precision/recall/F1 and thresholds; quantile/tail errors; gradient/spectral ratios; edge/terrain/coast errors; perturbation continuity; latency/RSS/artifact size.

## 29. Ablation plan

Run the consolidated experiment sequence in Appendix I. Required ablations: residual target, rollout curriculum, scheduled sampling, lead weights, direct multi-horizon challenger, static/geography features, one rain treatment, one sharpness term, proxy transport, and backbone challenger. An addition survives only if its intended metric improves without violating gates, cost, or other critical metrics.

## 30. Experiment management

Each run records run ID, hypothesis, config hash, code commit, source checksums, split, seed, environment, parameters, training budget, checkpoint rule, metrics, artifacts, latency/RAM, conclusion, and reviewer. Fixed naming: `E##_model_variant_seed_YYYYMMDD`. Raw results are immutable; summaries link to them. Locked test access is logged.

## 31. API specification

```mermaid
sequenceDiagram
  participant J as Judge
  participant S as serve.py
  participant M as Loaded engine
  J->>S: POST /health every 2 s (≤120 s)
  S-->>J: 200 {status: ready, model_name: ...}
  J->>S: POST /simulate nested physical lists
  S->>S: validate, copy, add delta once, normalize
  S->>M: rollout 1..14 steps
  M-->>S: normalized states
  S->>S: denormalize, semantic checks, serialize
  S-->>J: rollout + inference_ms (<5 s for 10)
```

The response is finite JSON. Canonical input/output is nested lists; b64 is tolerated but must not be relied on. No redirects. Error schema is not fully specified; use deterministic 4xx for malformed client input and fail visibly for internal errors rather than return fabricated forecasts.

## 32. Offline deployment design

```mermaid
flowchart TB
  subgraph C[Organizer container: network none, 4 CPU, 6 GB]
    subgraph RO[Read-only repository]
      S[serve.py]
      W[Weights + static maps + code]
    end
    TMP[/tmp only writable]
    S --> W
    S -. optional bounded temp .-> TMP
    J[Judge] -->|localhost/internal HTTP| S
  end
```

Load model/static assets once before ready; set thread counts; disable gradients; avoid runtime compilation/download; keep request state local; serialize float32-compatible finite lists. A public URL is unnecessary under default organizer execution.

## 33. Compute and resource budgets

Official limits versus internal targets (targets are PROPOSED):

| Resource | Official | Internal target |
|---|---:|---:|
| CPU | 4 cores | deterministic ≤4 threads |
| RAM | 6 GB | peak RSS ≤2 GB, stretch ≤3 GB |
| GPU | none | no GPU-only dependency/path |
| Startup | 120 s | ≤15 s p95 on representative CPU |
| 10-step wall time | <5.0 s | ≤2.0 s p95, ≤3.0 s max incl. JSON |
| Response | <80 MB | ~1 MB normal nested JSON |
| Repo/weights | under stated 2 GB | ≤200 MB total; model weights ≤50 MB preferred |
| Parameters | none stated | primary ≤5M; fallback ≤2M |

Measure preprocessing, model, postprocessing, and serialization separately and end-to-end. Estimates are not evidence until CPU benchmarks run.

## 34. Dashboard design

```mermaid
flowchart LR
  CTRL[Variable, horizon, region, shock controls] --> API[Live /simulate]
  API --> MAP[Prime/Multiverse maps]
  API --> AN[Difference/anomaly maps]
  API --> TS[Regional time series]
  API --> SD[Scientific diagnostics]
  API --> HR[Heat-risk proxy]
  LIM[Limitations/provenance] --> MAP
```

Layout: regional map centre; scenario controls left; lead/variable/time comparison top; anomaly/diagnostic/impact panels right/below; explicit model/data limitations always accessible. No chatbot or unrelated agent. Demonstration applies a new shock live and shows server-derived results.

## 35. Heat-risk impact module

Because `t2m` is a daily mean of four synoptic samples, output is a **temperature-anomaly / heat-risk proxy**, never an official heatwave declaration.

Training-only day-of-year baseline `C(d,cell)` and scale `S(d,cell)` may define `A=(T-C)/S` with smoothed seasonal estimates. A risk zone is a contiguous set exceeding approved absolute/anomaly thresholds for an approved persistence duration. Compare base vs perturbed area, duration, centroid, and intensity. Do not add population exposure, wet-bulb temperature, Tmax, health outcomes, or humidity claims not supported by inputs/methodology.

## 36. Repository architecture

```text
submission/
  serve.py                  # official entrypoint
  src/{data,features,models,physics,rollout,inference,evaluation}/
  training/                 # real reproducible training entrypoints
  configs/                  # immutable experiment/release configs
  artifacts/                # compact weights, static maps, manifests
  dashboard/                # separate live client
  tests/{unit,integration,scientific,contract,performance}/
  docs/                     # PRD, decisions, results, attribution
  scripts/                  # bounded train/evaluate/package commands
  README.md LICENSES.md THIRD_PARTY_NOTICES.md
```

Official reference materials and the 117 MB training dataset must not be blindly included in submission; keep them outside or gitignored with retrieval/provenance instructions consistent with rules. Git initialization is pending explicit approval.

## 37. Development methodology

Small reviewed increments; baseline always runnable; configuration over hidden constants; type/shape/unit assertions; deterministic seeds; pinned minimal dependencies; checkpoint manifests; automated contract/scientific regression; no major architecture change without ADR and ablation. Preserve organizer originals unchanged.

## 38. Team decomposition

Data/science, ML, systems/API, product/visualization, and reproducibility/review ownership are specified in Appendix J. Parallel work begins only after interface/constants are frozen; release owner arbitrates contract changes.

## 39. Implementation roadmap

P0 mandatory work establishes data/baselines/model/rollout/API/offline fallback/dashboard/release. P1 tests strong measured improvements. P2 research begins only after reliable core. P3 is future work. Dependencies, effort, risk, parallelism, and gates are in Appendix J.

## 40. Risk register

The probability/impact, warning, mitigation, fallback, and ownership matrix is in Appendix K. Highest risks are day-4 skill, drift/blur/extremes, shock fidelity, boundary artifacts, normalization mismatch, CPU latency, repo scanner/compliance, original-work provenance, and overclaiming counterfactual/heat output.

## 41. Failure-mode analysis

| Failure | Detection | Safe response / scientific impact |
|---|---|---|
| Nonfinite/negative rain/temp invalid | Per-step gate | Fail test/release; never disguise with persistence |
| Drift/scale collapse | Lead-wise bias/std/tails | Retrain/simplify; forecast scientifically unreliable |
| Blur or noise inflation | Variance + gradients/spectra + skill | Diagnose objective/backbone; reject gaming |
| Shock ignored/global/teleported | Causality suite | Reject candidate; no scripted spreading |
| Boundary reflection | Edge maps/perturbations | Change padding or disclose; reject severe behavior |
| Shape/unit/order mismatch | Sentinels/validator | Block release |
| Missing artifact/dependency | Clean clone/offline start | Package correctly or fallback |
| Timeout/RAM | external benchmark/RSS | Compact model/serialization; use validated fallback |
| Mutable server state | repeated/interleaved tests | Remove cache/state coupling |
| Dashboard fabrication | live provenance test | Block demo |

## 42. Acceptance gates

| Gate | Required evidence | No-go / fallback |
|---|---|---|
| Dataset readiness | Hash/schema/continuity/units/split tests | Stop training |
| Baseline readiness | Reproducible persistence/climatology/ridge report | Fix harness |
| Primary readiness | All validity, internal day-4 persistence win, stable leads, resource target trend | Simplify/use fallback |
| Novelty readiness | Predeclared ablation benefit across seeds and cost bound | Exclude feature |
| Perturbation readiness | Exact null + public causality + broad continuity/boundary suite | Reject model/extension |
| Generalization readiness | Locked chronology and strata without severe failure | Revisit model/claims |
| Deployment readiness | Validator/scanner/offline/read-only/4-core/6GB/latency passes | Optimize/fallback |
| Dashboard readiness | Live output, honest impact/limitations | Minimal compliant UI |
| Submission readiness | Clean clone, artifacts, brief, attribution, frozen tag | Do not submit |

No numerical architecture threshold beyond official gates is claimed before baselines establish realistic values.

## 43. Compliance checklist

Use the standalone checklist in Appendix L as the release gate. Every box needs a test artifact, log, hash, or reviewer sign-off; verbal assurance is insufficient.

## 44. Final submission strategy

Freeze a clean public repository with primary model selected by predeclared validation, independently working fallback, artifacts/checksums, training commands, separate live dashboard, two-page rendered/verified brief, licenses/notices, and evidence report. Rehearse from clean clone in network-none/read-only constraints. Record final tag/commit before deadline.

## 45. Scientific limitations and claim boundaries

- Daily coarse six-field state cannot reproduce full atmospheric dynamics or sub-daily extremes.
- No vertical profiles, lateral boundary conditions, sea-surface/soil/radiative/cloud state, or forcing.
- Regional moisture/energy are open-system; exact conservation claims invalid.
- Ten-metre wind is an incomplete transport proxy.
- Deterministic regression underrepresents uncertainty; any ensemble requires calibration.
- Reanalysis is not perfect observation.
- Artificial-shock plausibility is not causal truth.
- Heat proxy is not an official warning and lacks Tmax/population/health exposure.
- Performance applies only to measured splits/scenarios/hardware.

## 46. Future research opportunities

Probabilistic ensembles, uncertainty calibration, learned lateral boundary conditions, richer vertical/land/ocean state, conservation-aware closures with adequate variables, graph/mesh alternatives, sub-daily forecasting, formal counterfactual benchmarks, population/exposure coupling from permitted sources, and longer climate-distribution stability studies.

## 47. Unresolved questions

1. Does the presentation add/modify requirements? PDF/PPTX export is still inaccessible.
2. What exact organizer CPU/container image is used?
3. Does “under 2 GB” formally cover total repo checkout, weights, Git LFS objects, or submitted archive? Plan meets all interpretations where possible.
4. Team approval needed for split, primary/fallback, internal resource targets, heat-proxy methodology, and Git initialization.
5. Empirical questions—architecture, loss, curriculum, proxy physics, uncertainty—remain intentionally unresolved until experiments.

## 48. References

- Official problem statement, pages 1–2 and 7–10 (local verified PDF).
- [Official kit v1.0.0](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0) and exact [contract](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/blob/main/docs/R1_CONTRACT.md).
- Kochkov et al., [Neural general circulation models for weather and climate](https://www.nature.com/articles/s41586-024-07744-y), 2024.
- Lam et al., [GraphCast](https://doi.org/10.1126/science.adi2336) / current [WeatherNext repository](https://github.com/google-deepmind/weathernext).
- Bi et al., [Pangu-Weather](https://www.nature.com/articles/s41586-023-06185-3), 2023.
- Pathak et al., [FourCastNet repository](https://github.com/NVlabs/FourCastNet).
- Rasp et al., [WeatherBench 2](https://github.com/google-research/weatherbench2).
- Shi et al., [ConvLSTM](https://papers.nips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html), 2015.
- Li et al., [Fourier Neural Operator](https://arxiv.org/abs/2010.08895), 2020.

## Appendix A — dataset processing diagram

```mermaid
flowchart LR
  F[Verified NetCDF hash] --> X[xarray schema/metadata]
  X --> Q[Quality/range/chronology checks]
  Q --> S[Chronological split]
  S --> W[3-day input + 1..14 target windows]
  W --> N[Training-only/organizer transform policy]
  N --> L[Deterministic loaders]
```

## Appendix B — forecasting model diagram

```mermaid
flowchart LR
  D[3×6 normalized dynamics] --> C[Channel stack]
  G[LSM/orography/lat/lon/metrics] --> C
  S[DOY sin/cos] --> C
  C --> E[Native-scale encoder]
  E --> MS[Compact multiscale residual core]
  MS --> SK[Fine-scale skip fusion]
  SK --> H[6-channel tendency head]
  H --> U[Residual state update]
```

## Appendix C — physical/learned interaction diagram

```mermaid
flowchart LR
  X[Current state] --> FD[Metric finite differences]
  FD --> AP[Optional proxy advection/diffusion]
  X --> NN[Learned residual dynamics]
  G[Geography] --> NN
  AP --> NN
  AP --> SUM[Controlled tendency combination]
  NN --> SUM
  SUM --> NX[Next state]
```

## Appendix D — evaluation diagram

```mermaid
flowchart TD
  CP[Candidate prediction] --> V{Validity gate}
  V -->|fail| Z[Zero for failed window / no release]
  V -->|pass| FS[Forecast skill 45]
  V --> DY[Dynamics 25]
  FS --> RM[RMSE skill 30]
  FS --> CSI[Rain CSI 15]
  DY --> AB[Anti-blur 10]
  DY --> CA[Causality 15]
  RS[Repo scanner 15] --> TOT[Total + brief 15]
  RM & CSI & AB & CA --> TOT
```

## Appendix E — final self-review summary

Atmospheric-science review removed exact-conservation/full-physics claims and qualified heat output. ML review added chronological isolation, per-seed ablations, baseline-relative metrics, and simple-first losses. Systems review separated judge wall time from `inference_ms`, added serialization/RSS/read-only/offline budgets, and kept a trained fallback. Software review froze unit/shape/perturbation interfaces and modular ownership. Evaluator review reconciled pages 1/7–10 with contract/scoring/scanner and captured max steps 14 plus exact thresholds. Skeptical review rejected direct global checkpoints, mandatory proxy physics, excessive novelty claims, hard clipping, and decorative features. Remaining uncertainty is presentation content and all future empirical performance.

## Appendix F — consolidated source inventory and integrity record

Evidence labels: **OFFICIAL** means inspected organizer primary material; **DATASET** means measured from the checksum-verified artifact; **SCIENCE** means an external paper/upstream repository; **TEAM** means a direction or note supplied by the team.

| Resource | Version / integrity | Evidence and use |
|---|---|---|
| Problem statement PDF | SHA-256 `0310be472b37b31d591e5f9f44654137b92eacd0ffdb941366a06a9ef4d9ff77` | OFFICIAL; rendered/inspected pages 1–2 and 7–10 |
| Official kit repository | tag `v1.0.0`, commit `689c8798bbff7f318a1e74878a6985abc646080a` | OFFICIAL; exact contract, scoring, validator, scanner, starter, tooling |
| Dataset | 122,831,709 bytes; SHA-256 `f37bdbd0bd44e6fe951adf36c9a08620075f9d17de391b04c97d61f06797903b` | OFFICIAL/DATASET; xarray audit completed |
| `stats.json` | SHA-256 `f00103a38a93dcbaea4e4bd31da3db6862812cd2807bd2224c51264054e014a7` | OFFICIAL; six means/stds |
| Release contract | SHA-256 `e49e70c065d3b709ade37718ae2466a8e4845daaa4603b7d6793b209a3db5513` | OFFICIAL; exact API/scoring addendum |
| Kit wheel | SHA-256 `23c983b8be2715af9b472be1a74a137820e91f3e9bd21755af450ada25000672` | OFFICIAL; version/dependencies/contents inspected |
| License/notice | MIT code; modified ERA5 via WeatherBench 2, ERA5 CC BY 4.0 | OFFICIAL; attribution required |
| Organizer presentation | Canva URL only; export inaccessible | OPEN; compare PDF/PPTX when obtained |
| Team briefs/directives | unversioned supplied attachments | TEAM; project philosophy and approval process |

All manifest-listed release artifacts matched their organizer SHA-256 values. The official clone remains separate from the future submission repository and clean at the tagged commit.

## Appendix G — consolidated mandatory requirements and verification matrix

| ID | Mandatory requirement | Primary source | Verification mechanism | Failure consequence |
|---|---|---|---|---|
| API-01 | `POST /health` returns 200 ready/model object | Contract L21–22 | Official validator and exact schema assertion | Evaluator never starts |
| API-02 | `POST /simulate`; three oldest-first states; physical units | Contract L24–33 | Temporal/channel/unit sentinels | Invalid forecast/scoring |
| API-03 | Shapes `[3,6,32,36]`, nullable delta `[6,32,36]`, output `[steps,6,32,36]` | Contract L24–31 | Shape/property tests | Validity failure |
| API-04 | Channel order `t2m,tp,u10,v10,msl,tcwv` | Contract L9–18 | Per-channel sentinel | Catastrophic score error |
| API-05 | Apply physical perturbation once to latest state | Contract L28/33 | No-repeat and no-input-mutation tests | Shock invalid |
| API-06 | `doy_sincos` is last input day; denominator 365 | Contract L28/37 | Known-day tests and analytic advancement | Seasonal misalignment |
| API-07 | Finite nested lists canonical; b64 tolerated only | Contract L31 | Validator both formats; canonical response | Schema failure |
| API-08 | Support steps 1–14; scored at 10 | `contract.py` | Bounds and shape tests | Hidden-valid-call failure |
| API-09 | Actual `inference_ms`; response <80 MB; no redirects | Contract L29/42 | Timing/HTTP integration | Contract failure |
| DATA-01 | Use official 2005–2021 daily six+two-channel artifact | PS p7/p9; release | Hash and schema | Wrong training basis |
| DATA-02 | Preserve ascending 6.5–37.5°N, 65.5–100.5°E grid | Contract L5–7 | Exact coordinate/corner tests | Spatial inversion |
| DATA-03 | Bundle static fields; API will not send them | Contract L35 | Clean packaged inference | Runtime failure |
| SCI-01 | Valid finite rollout; rain nonnegative; temperature 233.15–333.15 K | Contract/scoring | Exact public validity function | Hard-gate failure |
| SCI-02 | Beat persistence at day 4 | PS p10; contract L53 | Locked-test official-style skill | Does not qualify |
| SCI-03 | Skill vs stronger persistence/climatology at days 1/4/8 | Contract L51 | Metric parity tests | Forecast-point loss |
| SCI-04 | Heavy-rain CSI >10 mm/day at days 1/4/8 | Contract L51 | Exact CSI implementation | Lose rain points |
| SCI-05 | Day-8 variance ratio 0.40–2.50 per channel | Contract L52 | Exact ratio tests plus anti-noise diagnostics | Lose anti-blur points |
| SCI-06 | Shock nonzero/local/propagating/stable; zero within tolerance | Contract L46–53 | Public causality plus broad matrix | Lose causality/gate |
| NFR-01 | Offline, no runtime downloads/APIs | PS p8/p10; contract L39 | Network-none clean clone | Disqualification/failure |
| NFR-02 | 4 CPU, 6 GB, no GPU; ten steps <5 s | PS p8; contract L39/42 | Constrained end-to-end benchmark | Timeout/resource failure |
| NFR-03 | Read-only repository; `/tmp` only | PS p8 | Read-only mount/write trace | Sandbox failure |
| NFR-04 | Ready ≤120 s; eight back-to-back calls, no retry | Contract L39 | Cold/warm/soak suite | No score or failed calls |
| NFR-05 | Root `serve.py`, bind `0.0.0.0` | PS p8; contract L39 | Clean launch/socket probe | Unreachable server |
| COMP-01 | Public repo created during hackathon; original work | PS p1/p10 | Git/provenance review | Disqualification |
| COMP-02 | Training, weights, root server, separate dashboard, README | PS p8 | Scanner/manual clean-clone review | Repo-point loss |
| COMP-03 | Local weights; stated under-2-GB condition | PS p8 | Packaged-size audit | Packaging failure |
| COMP-04 | No dates/lookups/weather archives/pre-rendered demo | PS p10; scanner | Official scanner/import/manual review | Disqualification |
| COMP-05 | Two-page prescribed brief | PS p1/p8 | Render and content review | Brief-point loss |

Interpretation reconciliations: contract-supplied `doy_sincos` is permitted while date lookup is prohibited; heat-risk is a selected extension; organizer-run offline execution is the default; validator success does not establish forecast skill; code fixes maximum steps at 14 while ten are scored.

## Appendix H — consolidated dataset statistics

Structure: `data[6209,6,32,36]` float32, `static[2,32,36]` float32; continuous 2005-01-01 through 2021-12-31; zero missing days, duplicates, NaNs, or infinities.

| Channel | Unit | Min | Mean | Std | Median | P95 | P99 | Max | Lag-1 corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t2m | K | 236.327 | 292.357 | 13.464 | 299.225 | 303.997 | 307.932 | 314.902 | .997 |
| tp | mm/day | 0 | 3.618 | 7.553 | .551 | 17.182 | 35.701 | 249.774 | .712 |
| u10 | m/s | -16.621 | .888 | 2.899 | .423 | 6.562 | 9.347 | 17.754 | .936 |
| v10 | m/s | -16.999 | .191 | 2.427 | .089 | 4.849 | 7.237 | 17.692 | .907 |
| msl | Pa | 97,877.8 | 101,122.3 | 619.019 | 101,072.7 | 102,254.8 | 102,925.0 | 105,115.9 | .947 |
| tcwv | kg/m² | .169 | 30.034 | 20.494 | 27.487 | 61.417 | 67.030 | 79.545 | .985 |

- Precipitation has 7.81% exact zeros and P99.9 71.81 mm/day.
- Adjacent spatial correlations range approximately .90–.99; precipitation is least smooth.
- Land-sea mask is fractional in 918/1,152 cells; orography ranges -0.156 to 5,283.94 m.
- Land-dominant/ocean-dominant mean `tcwv` is 22.31/43.69 kg/m²; `tp` 3.03/4.66 mm/day.
- East-west physical cell width varies approximately 110.5→88.2 km; gradients require latitude-aware `dx`.
- Strong seasonality: domain rain mean rises from 1.03 mm/day in January to 7.77 in July; `tcwv` 17.84→44.26 kg/m².
- `t2m` is daily mean of four synoptic samples, not daily maximum.

Proposed split: train 2005–2017, validate 2018–2019, lock 2020–2021; no window may cross partitions. This remains an approval decision.

## Appendix I — consolidated experiment and test program

### Experiment sequence

| ID | Controlled question | Acceptance / decision |
|---|---|---|
| E00 | Data, shapes, dates, transforms | Exact sentinel agreement; blocks all training if failed |
| E01 | Persistence, seasonal climatology, ridge | Complete reproducible lead-wise baseline |
| E02 | Direct state vs residual tendency | Select using day 1/4/8/10 and drift, not day 1 alone |
| E03 | One-step vs 2→4→8→10 curriculum | Improve days 8/10 in ≥2/3 seeds; ≤2% proposed day-1 regression tolerance |
| E03A | Teacher forcing vs scheduled sampling vs self-fed | Robust exposure-gap reduction; no shock instability |
| E03B | Uniform vs increasing lead weights | Late gain without short-lead/tail sacrifice |
| E03C | Autoregressive vs matched direct multi-horizon | Retain only a Pareto improvement with temporal consistency |
| E04 | Dynamic only vs static vs coordinate/metric features | Overall plus edge/terrain benefit |
| E05 | Base vs one rain-tail objective | Better CSI without >2% aggregate RMSE loss or false-alarm surge |
| E06 | Base vs one gradient/sharpness objective | Better structure without noise/RMSE regression |
| E07 | Learned-only vs proxy transport | Skill/causality gain with <10% latency overhead |
| E08 | Geo-CNN vs ConvLSTM/small operator | Pareto comparison only after stable primary |
| E09 | Broad unfamiliar perturbation matrix | Finite, bounded, continuous, local, zero-consistent |
| E10 | Primary and fallback packaging | Validator/scanner/offline/read-only/resource gates |

### Mandatory test layers

- Unit: normalization round trip, residual update, metrics, padding, perturb-once, rain semantics.
- Data: hashes, axes, dates, split isolation, target alignment.
- Model properties: finite/shape/determinism/no mutation/null equality.
- Scientific regression: per-channel/region days 1/4/8/10, CSI, tails, variance, gradients, spectra, drift.
- Perturbation: all channels, signs, magnitude sweep, shapes, land/ocean/coast/terrain/boundaries.
- API: official validator, malformed payloads, steps 1/10/14 and invalid bounds.
- Deployment: 4 cores, 6 GB, network none, read-only mount, cold start, repeated calls.
- Repository: unmodified scanner, import closure, artifact provenance, no archives/APIs/dates.

Every retained experiment records config/code/data hashes, split, seed, schedule, horizon, lead weights, parameters, training budget, checkpoint rule, per-lead/channel/region metrics, perturbation results, latency/RSS/size, and retain/revise/abandon conclusion.

## Appendix J — consolidated implementation roadmap and ownership

No task is authorized until architectural and Git approval.

| Task | Priority | Depends on | Deliverable / gate | Effort |
|---|---|---|---|---:|
| Separate submission repo and provenance | P0 | Approval | Clean non-reference repo and ignores | 0.5 d |
| Data loader/window/split tests | P0 | Split approval | E00 pass | 1 d |
| Baselines and exact metrics | P0 | Data layer | E01 report | 1 d |
| Minimal residual CNN | P0 | Data layer | Reproducible checkpoint | 1–2 d |
| Shared rollout and perturb-once | P0 | Minimal model | Null/stability/property pass | 1 d |
| API integration | P0 | Rollout | Validator pass | 0.5 d |
| Offline/resource hardening | P0 | API | All sandbox gates | 1 d |
| Independently trained fallback | P0 | Baseline model | Separate validated artifact | 0.5–1 d |
| Minimal live dashboard/impact proxy | P0 | Stable API | Live-output provenance | 1–2 d |
| README/brief/attribution/release | P0 | All P0 | Clean release candidate | 1 d |
| Multi-step/sampling/lead experiments | P1 | Stable baseline | E03 decisions | 1–3 d |
| Geography/rain/sharpness experiments | P1 | Stable baseline | E04–E06 decisions | 1–3 d |
| Proxy physics/backbone challenger | P2 | Strong P1 | E07–E08 decisions | 2–3 d each |
| Probabilistic/boundary-rich research | P3 | Post-hackathon evidence/data | Future research | weeks+ |

Ownership: atmospheric/data lead owns physical interpretation and strata; ML lead owns baselines/training/ablations; systems lead owns API/offline/performance; product lead owns live visualization/impact; reproducibility owner owns registry, compliance, and release review. Parallel work begins only after constants/interfaces are frozen.

## Appendix K — consolidated risk and assumption register

| Risk | P/I | Warning/detection | Mitigation/fallback |
|---|---|---|---|
| Fails day-4 persistence | M/H | Nonpositive locked skill | Residual/curriculum/simpler stable fallback |
| Long-horizon drift/explosion | M/H | Rapid error/scale growth | Multi-step exposure, tendency scaling, rollout gate |
| Blur/extreme suppression | H/H | Low variance/gradients/CSI | One targeted loss; joint sharpness and skill review |
| Noise used to pass variance | L/H | Variance passes while spectra/RMSE fail | Joint spectra/gradients/skill; reject gaming |
| Shock ignored/global/teleported | M/H | Public/broad causality failures | Shared local dynamics; reject scripted spread |
| Boundary artifacts | H/M | Edge error/reflection | Nonperiodic padding ablation; disclose/reject severe |
| Normalization/channel/unit mismatch | M/H | Sentinel/round-trip failure | Single transform/constants source; release block |
| Latency/RAM/startup failure | M/H | Constrained benchmark miss | Compact model/serialization; validated fallback |
| Offline/read-only dependency failure | M/H | Network/write attempt | Network-none/read-only CI; bundle all assets |
| Scanner/compliance/originality failure | M/H | Official scan/history issues | Scan every release; provenance and exclusion audit |
| Dashboard not live | M/H | Fixtures/cached path | API provenance integration test |
| Heat/counterfactual overclaim | M/H | Unsupported terminology | Qualified proxy language and scientific sign-off |
| P2 schedule overrun | H/M | No result by gate | Abandon experimental component; preserve P0 |
| Canva contains extra rule | M/M | Export discrepancy | Obtain export; formally reconcile |

Key assumptions: kit v1.0.0 remains current; supplied seasonal pair is allowed; impact choice is flexible; internal split is team-selected; daily mean temperature cannot establish official heatwaves; development CPU timing needs margin because exact judge hardware is unknown.

## Appendix L — standalone final compliance checklist

### Repository and artifacts

- [ ] Public submission repo created during permitted hackathon period, separate from organizer references.
- [ ] Root `serve.py`, real training pipeline, included weights, separate dashboard, README.
- [ ] Packaged size satisfies the stated under-2-GB condition; exact size/hash recorded.
- [ ] Dataset/multi-year weather archives excluded; required static maps included.
- [ ] MIT/ERA5/WeatherBench and all third-party attribution included.
- [ ] Two-page brief rendered and checked: model/drift/blur/failures on page 1; visualization/impact on page 2.

### API and scientific behavior

- [ ] Exact channel order, shapes, physical units, oldest-first history, and seasonal semantics.
- [ ] Perturbation copied and applied exactly once; request input never mutated.
- [ ] Steps 1, 10, and 14 pass; invalid bounds handled deterministically.
- [ ] Finite output; rain nonnegative; temperature validity range respected.
- [ ] Official validator passes canonical lists and tolerated b64 cases.
- [ ] Day-4 persistence hard gate passed on locked internal test.
- [ ] Days 1/4/8/10 reported per channel; heavy-rain/extreme and drift gates passed.
- [ ] Day-8 spatial variance range met without noise inflation.
- [ ] Null tolerance and broad unfamiliar-shock stability/locality/continuity passed.

### Sandbox and release

- [ ] `python serve.py --port 8000`, bind `0.0.0.0`, ready ≤120 s.
- [ ] Ten steps <5 s end-to-end; response <80 MB; 4 cores; RAM <6 GB; no GPU.
- [ ] Network-disabled and read-only repository execution; writes only under `/tmp`.
- [ ] Stateless repeated/interleaved calls; primary and fallback independently work.
- [ ] Official scanner has no unresolved critical flags; warnings reviewed.
- [ ] Dashboard and impact panels consume live `/simulate`, never pre-rendered output.
- [ ] Clean-clone rehearsal, artifact hashes, final tag/commit, and deadline freeze completed.
- [ ] Scientific claims and limitations approved by atmospheric-science reviewer.

## Appendix M — decisions requiring explicit team approval

Before implementation, approve or revise:

1. Proposed chronological split: 2005–2017 / 2018–2019 / locked 2020–2021.
2. Geography-aware residual CNN as primary candidate and simpler residual CNN as fallback.
3. Autoregressive shared transition as default, with a mandatory direct multi-horizon challenger experiment.
4. Multi-step curriculum/scheduled-sampling/lead-weight ablation plan and drift acceptance tolerances.
5. Proxy transport as P2 experimental only, with strict abandon criteria.
6. Internal resource targets and maximum preferred parameter/artifact sizes.
7. “Temperature-anomaly / heat-risk proxy” as impact terminology and methodology direction.
8. Creation and initialization of a separate submission Git repository.

Approval of this PRD authorizes only the agreed next phase; it does not convert hypotheses or estimates into validated results.


