# Deadlock — Multiverse of Monsoon

Submission repository for a compact, physics-aware regional atmospheric forecaster. The current code is the P0 contract foundation: a deterministic persistence baseline behind the exact organizer API. It is intentionally simple and will remain runnable while learned candidates are developed.

Team continuation starts with [`docs/HANDOFF.md`](docs/HANDOFF.md). The complete
requirements, architecture, acceptance gates, and experiment ledger are in
[`docs/PRODUCT_REQUIREMENTS.md`](docs/PRODUCT_REQUIREMENTS.md).

## Current state

- Root `serve.py` implements `/health` and `/simulate`.
- Inputs are copied and validated in physical units.
- Perturbations are applied exactly once to the latest input day.
- Every integer horizon from 1 through 14 is supported.
- The baseline is deterministic, stateless, finite, and safe for repeated calls.
- The verified training reader and baseline evaluator produce days 1/4/8/10 evidence.
- The learned-only GRAST configuration passed the frozen validation gates for three scaled seeds; seed 17 is the provisional primary candidate and is not served yet.
- Retained reports and the seed-17 state dict are committed for team continuation; generated sweeps remain excluded.
- No dataset or organizer repository is committed here.

## Local setup

```bash
python -m pip install -e ".[dev]"
pytest
python serve.py --port 8000
```

The official schema validator should be run against `http://localhost:8000` after installing the organizer kit separately. Training-only dependencies are available through `.[train]` and are not required by the judge-facing server.

## Baseline evaluation

Keep the official NetCDF outside this repository, then run:

```bash
python scripts/evaluate_baselines.py /path/to/india_climate_train.nc --partition validation --output reports/baselines/validation.json
```

The reader verifies the official SHA-256, full schema, coordinates, channel order, units, finiteness, and rain semantics before evaluation. Calendar climatology and model transforms fit the training partition only. The report covers persistence and climatology at days 1, 4, 8, and 10 with latitude-weighted RMSE, heavy-rain CSI/precision/recall/false alarms, and spatial-variance ratios.

For a bounded real-data optimization smoke test:

```bash
python scripts/prototype_train.py /path/to/india_climate_train.nc --output reports/training/prototype_seed7.json --windows 64 --epochs 10 --horizon 2
```

This prototype proves that the shared rollout trains and preserves basic physical validity. It is deliberately labelled non-release evidence and does not establish held-out forecast skill.

Chronological candidate training is available through `scripts/train_candidate.py`. Generated reports and checkpoints are ignored by Git until a release candidate passes physical validity, day-8 anti-blur, heavy-rain activity, and rollout-aware days 1/4/8/10 selection gates.

## Architecture progression

1. Keep this persistence engine as the always-runnable contract baseline.
2. Add leakage-free data windows and train-partition transforms.
3. Add learned-only GRAST with shared autoregressive rollout and multi-step training.
4. Time-box the bounded transport/residual gate against learned-only and additive controls.
5. Export primary and independent fallback artifacts without runtime downloads.

The repository contains the authoritative PRD, retained three-seed evidence,
and the provisional candidate state dict. The organizer dataset, failed sweep
artifacts, and redundant checkpoints remain excluded.

