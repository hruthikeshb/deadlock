# P0 workstreams

The repository is split by stable interfaces so tracks can progress in parallel without duplicating contract logic.

| Track | Owns | First gate | Must not own |
|---|---|---|---|
| Data and science | NetCDF reader, chronological windows, train-only statistics, units, metrics | Dataset sentinels and leakage tests | HTTP or model architecture |
| ML and rollout | Baselines, GRAST, multi-step curriculum, checkpoint export | Reproducible days 1/4/8/10 baseline report | Wire parsing or dashboard |
| API and deployment | Request validation, engine loading, `/health`, `/simulate`, CPU benchmarks | Official schema validator with persistence engine | Training transforms |
| QA and release | Property tests, evidence manifests, clean-clone/offline/read-only rehearsal | P0 regression suite | Silent threshold changes |
| Product | Live API dashboard and qualified diagnostics | Provenance test proving live `/simulate` data | Pre-rendered forecasts |

## Shared interface freeze

- Dynamic channel order: `t2m, tp, u10, v10, msl, tcwv`.
- Input: float32 `[3, 6, 32, 36]`, oldest first, physical units.
- Perturbation: optional float32 `[6, 32, 36]` physical delta, copied and applied once.
- Output: float32 `[steps, 6, 32, 36]`, physical units, finite.
- Steps: every integer from 1 through 14.
- Engines implement `simulate(init_states, doy_sincos, steps, perturbation)` and may not inspect named scenarios.
- Model normalization uses training-partition statistics only. Organizer statistics remain evaluator references.

## Immediate sequence

1. **Complete:** contract baseline and tests so all later models inherit one serving path.
2. **Complete:** verified dataset loader, split/window index, and train-only transform foundation.
3. **Complete:** persistence and calendar-day climatology evaluation at days 1/4/8/10.
4. **Complete as a tested model seam:** learned-only residual transition, semantic rain/temperature mapping, seasonal advancement, and self-fed rollout.
5. **In progress:** rollout-aware loss, curriculum, artifact manifests, real-data windows, and a two-lead training prototype are complete; chronological validation/checkpoint selection and independently loadable exports are next.
6. **Later P1:** three-hour maximum gated-transport experiment; gate off unless it clearly earns retention.

