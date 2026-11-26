# Engineering handoff

Branch: `feat/p0-foundation`  
Authoritative requirements: `docs/PRODUCT_REQUIREMENTS.md`

## Current verified state

- Official dataset SHA-256: `f37bdbd0bd44e6fe951adf36c9a08620075f9d17de391b04c97d61f06797903b`.
- The organizer dataset is deliberately excluded from Git. Obtain it from the official v1.0.0 release and verify it with `scripts/audit_precipitation.py`.
- Chronological isolation is fixed: train 2005–2017, validation 2018–2019, locked test 2020–2021.
- Frozen GRAST configuration passed all current validation gates for seeds 7, 17, and 29 without retuning.
- Seed 17 is the provisional primary candidate. Its raw state dict is `artifacts/candidate/grast_seed17.pt` (SHA-256 `711b7287dc4df52f609d35e674cce8e6e44d2bdccc008845eb92e0d8036a4d2d`); config and evaluation evidence are in `evidence/training/scaled_seed17.json`.
- `artifacts/primary` is the hash-verified, self-describing runtime bundle. `serve.py` now loads it as the primary engine and falls back to persistence if verification/loading fails.
- `evidence/diagnostics/perturbation_seed17_summary.json` records the passing 370-scenario validation-only unfamiliar-shock matrix.
- `evidence/release/official_checks.json` records the official validator pass against the learned engine and the flag-free 14/15 repository scan.
- The locked test has not been used for checkpoint selection or evaluation.
- Run `python -m pytest -q`; the expected result at handoff is 59 passing tests.

## Completed

- Official API contract, list/base64 codec, perturb-once semantics, physical validity, persistence fallback, and validator coverage.
- Verified NetCDF reader, exact hash/schema/coordinate/unit gates, chronological windows, and train-only normalization.
- Persistence/climatology baselines and days 1/4/8/10 metrics.
- Multiscale GRAST core, shared autoregressive rollout, 2→4→8→10 curriculum, rain occurrence/intensity head, and train-prior calibration.
- Geography-feature ablation and perturbation/error diagnostic primitives.
- Three-seed scaled validation confirmation. Aggregate normalized RMSE is 0.5244/0.5095/0.5323 for seeds 7/17/29.

## Remaining work in order

1. Export and verify an independently usable fallback bundle (the persistence runtime fallback already remains available).
2. Benchmark explicit steps 1/10/14, cold start, peak RSS, response size, eight-call soak, network-disabled execution, and read-only repository behavior on four CPU cores.
3. Only after items 1–2 pass, perform the one-time locked 2020–2021 test evaluation and freeze the candidate. Do not tune after viewing locked-test results.
4. Complete attribution and README commands, then run a clean-clone release rehearsal.

## Explicitly excluded from this workstream

- Dashboard/frontend implementation (the one currently missing repository-scanner point).
- Two-page competition brief/presentation production.

These were removed from the active implementation scope by team direction on 2026-11-26. They may be completed separately by teammates, but this workstream will not spend engineering time on them.

## Stop rules

- Do not add the official dataset or failed sweep checkpoints to Git.
- Do not enable optional transport physics before the learned-only candidate passes packaging, perturbation, and runtime gates.
- Do not relax false-alarm, anti-blur, physical-validity, or day-4 persistence gates.
- Do not use post-hoc clipping, injected noise, validation-derived calibration, or test-set tuning.
