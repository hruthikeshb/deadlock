# 🌦️ Multiverse of Monsoon

> **Track 2 — Yug Hackathon · Earth-System Modelling & Spatiotemporal Deep Learning**
>
> A fast, physics-aware **Climate Digital Twin** of India's atmosphere — simulates weather forward in time and branches into alternate universes when you apply localized shocks like a hotter Bay of Bengal or a weakened Arabian Sea monsoon wind.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference API](#-inference-api)
- [Channels & Units](#-channels--units)
- [Benchmark Results](#-benchmark-results)
- [Constraints & Limits](#-constraints--limits)

---

## 🌏 Overview

Standard weather apps answer **"Will it rain tomorrow?"**
This system answers **"How does India's atmosphere evolve — and what happens if we change the starting conditions?"**

The model takes the last **3 days** of India's atmospheric state across **6 channels** on a `32 × 36` grid (6.5°–37.5°N, 65.5°–100.5°E) and rolls the weather forward **day by day** for up to 10 steps. Apply a **perturbation** (physical delta added to the last input day) to branch into an alternate timeline.

**Key design goals:**
- 🚫 No NaN / Inf across any rollout
- 🌧️ Precipitation always ≥ 0 (physical clamping at every step)
- 📈 Beat "repeat yesterday" and seasonal climatology at Day 4
- 🗺️ Localized shocks stay near their origin at Day 1 and spread outward naturally
- ⚡ 10-step rollout completes in **< 5 seconds** on 4 CPU cores / 6 GB RAM (achieved: ~580 ms mean)

---

## 🧠 Architecture

### GRAST — Geography-aware Residual Atmospheric State Transition

```
Input [B, 26, 32, 36]
  = 3 × 6 dynamic history  (normalized)
  + 6 geo features          (LSM, orography, lat, lon, cos_lat, dx_km)
  + 2 seasonal features     (sin/cos of day-of-year, analytically advanced each step)

Encoder ──► Fine ResBlocks ──► DownBlock (stride-2)
                │                    │
                │              Coarse ResBlocks
                │                    │
                └────── UpBlock (bilinear + skip concat) ──► Fine ResBlocks ──► Head
                                                                                  │
                                                                           Tendency Δz [B, 6, 32, 36]
                                                                        × learnable residual scale r

Output: z_{t+1} = z_{t} + r ⊙ Δz    (applied in normalized space)
        x_{t+1} = denormalize(z_{t+1})
        tp, tcwv clamped ≥ 0 in physical space
```

**Anti-blur mechanisms:**
- `NonPeriodicPad` (replicate) — prevents artificial periodic boundary wrapping
- Learnable per-channel **residual scale** `r` (initialized to 0.1) — controls step size conservatively
- Multi-horizon **curriculum**: trained sequentially at horizon 1 → 2 → 4 → 8
- **Huber loss** (δ=1.0) in normalized space — robust to heavy-rain outliers

| Property | Value |
|---|---|
| Parameters | **1,402,220** (~1.4M) |
| Base channels | 64 |
| Training horizon | 8 steps |
| Input channels | 26 (18 dynamic + 6 geo + 2 seasonal) |
| Output | 6-channel atmospheric tendency |

---

## 📁 Repository Structure

```
.
├── serve.py                    # 🚀 Inference server (FastAPI) — run this for evaluation
├── requirements.txt            # Pinned dependencies
│
├── artifacts/
│   ├── primary_grast.pt        # ✅ Trained model weights (plain git, ~5.4 MB)
│   ├── training_log.jsonl      # Per-epoch training metrics
│   └── benchmark_report.json  # Full evaluation report (locked test 2020–2021)
│
├── data/
│   ├── india_climate_train.nc  # Daily climate data 2005–2021 (32×36 grid)
│   └── stats.json              # Channel-wise mean & std (used only inside the model)
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # DatasetReader, Transform, WindowIndex, splits
│   │   ├── hdf5_reader.py      # NetCDF loader
│   │   └── torch_dataset.py    # PyTorch Dataset wrapper
│   ├── features/
│   │   └── geo.py              # Static geo-feature builder (LSM, oro, lat/lon, cos_lat, dx)
│   ├── models/
│   │   └── grast.py            # GRAST + FallbackCNN model definitions
│   └── rollout/
│       ├── engine.py           # RolloutEngine (inference) + differentiable_rollout (training)
│       └── loss.py             # MultiStepHuberLoss
│
├── training/
│   ├── train.py                # Training entrypoint
│   ├── evaluate.py             # Comprehensive evaluation suite
│   └── kaggle_train.ipynb      # Kaggle notebook version
│
└── tests/
    ├── benchmark_metrics.py    # Stage-1 automated benchmark (5 regimes + 3 shocks)
    └── dataset_test.py         # Dataset integrity checks
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the inference server

```bash
python serve.py --port 8000
```

The server loads weights from `artifacts/primary_grast.pt` once at startup and listens on `0.0.0.0:8000`.

### 3. Validate with the kit validator

```bash
pip install monsoon_r1_kit-1.0.0-py3-none-any.whl   # from kit release
python -m monsoon_r1.schema_validator http://localhost:8000
```

Expected output:
```
health: OK
  [JSON lists (canonical)] no perturbation  OK   ~2.5s
  [JSON lists (canonical)] warm_ocean       OK   ~2.5s
  [JSON lists (canonical)] zeros            OK   ~2.5s
  [base64 (also tolerated)] no perturbation  OK   ~2.5s
  [base64 (also tolerated)] warm_ocean       OK   ~2.5s
  [base64 (also tolerated)] zeros            OK   ~2.5s
RESULT: all checks passed
```

---

## 🏋️ Training

### Full training pipeline (progressive curriculum)

```bash
# Phase 1: 1-step warm-up
python training/train.py --model primary --horizon 1 --epochs 40

# Phase 2: 2-step
python training/train.py --model primary --horizon 2 --epochs 30

# Phase 3: 4-step
python training/train.py --model primary --horizon 4 --epochs 20

# Phase 4: 8-step fine-tune
python training/train.py --model primary --horizon 8 --epochs 15
```

Each phase resumes from the previous checkpoint in `artifacts/primary_grast.pt`. Training logs are appended to `artifacts/training_log.jsonl`.

### Train the fallback CNN

```bash
python training/train.py --model fallback --horizon 4 --epochs 40
```

### Data splits (chronological, no leakage)

| Split | Date Range | Days |
|---|---|---|
| Train | 2005-01-01 → 2017-12-31 | 4748 |
| Val | 2018-01-01 → 2019-12-31 | 730 |
| Test (locked) | 2020-01-01 → 2021-12-31 | 731 |

---

## 📊 Evaluation

### Full offline evaluation (locked test 2020–2021)

```bash
python training/evaluate.py --checkpoint artifacts/primary_grast.pt
```

### Stage-1 automated benchmark (all 5 regimes + 3 shock types)

```bash
python tests/benchmark_metrics.py --checkpoint artifacts/primary_grast.pt
```

### Dataset integrity check

```bash
python tests/dataset_test.py
```

---

## 🔌 Inference API

The server exposes two endpoints:

### `POST /health`
```json
{ "status": "ready", "model_name": "primary_grast" }
```

### `POST /simulate`

**Request:**
```json
{
  "init_states":   "[3, 6, 32, 36] nested list — 3 input days, physical units",
  "doy_sincos":    [0.42, -0.91],
  "steps":         10,
  "perturbation":  "[6, 32, 36] physical delta added to init_states[-1], or null"
}
```

**Response:**
```json
{
  "rollout":      "[steps, 6, 32, 36] nested list — physical units",
  "inference_ms": 580.2
}
```

> `stats.json` is used **only inside** the model for normalization. All inputs and outputs use **raw physical units**.

---

## 📡 Channels & Units

| Index | Channel | Unit | Description |
|---|---|---|---|
| 0 | `t2m` | K | 2-metre air temperature |
| 1 | `tp` | mm/day | Total precipitation (always ≥ 0) |
| 2 | `u10` | m/s | 10-metre eastward wind |
| 3 | `v10` | m/s | 10-metre northward wind |
| 4 | `msl` | Pa | Mean sea-level pressure |
| 5 | `tcwv` | kg/m² | Total column water vapour (always ≥ 0) |

**Static fields** (bundled in `data/india_climate_train.nc`, loaded at startup):

| Field | Description |
|---|---|
| `land_sea_mask` | Fractional LSM 0–1 |
| `orography_m` | Surface elevation in metres |

---

## 🏆 Benchmark Results

Evaluated on the **locked test split (2020–2021)**, `n_windows = 719`.

### Forecast Skill — Skill Score vs best of (Persistence, Climatology)

| Lead | `t2m` | `tp` | `u10` | `v10` | `msl` | `tcwv` | **Mean** |
|---|---|---|---|---|---|---|---|
| Day 1 | +0.146 | +0.232 | +0.229 | +0.272 | +0.310 | +0.235 | **+0.237** |
| Day 4 | +0.081 | +0.067 | +0.125 | +0.104 | +0.121 | +0.139 | **+0.106** |
| Day 8 | -0.079 | +0.016 | -0.006 | -0.013 | +0.021 | -0.019 | **-0.013** |

### Heavy Rain Detection (Day 1, CSI > 10 mm/day)

| Model | CSI | POD | Precision | FAR |
|---|---|---|---|---|
| **GRAST** | **0.476** | 0.557 | **0.766** | 0.234 |
| Persistence | 0.429 | 0.601 | 0.601 | 0.400 |
| Climatology | 0.263 | 0.399 | 0.435 | 0.565 |

### Causal Shock Tests

| Shock | Day-1 Local Energy | Spread by Day 4 | Finite & Stable | Pass |
|---|---|---|---|---|
| Zero shock | — | — | ✅ 0.0 norm diff | ✅ |
| Warm Bay of Bengal (+2 K) | 98.5% | 14.1% outside | ✅ Day-10 max 0.72σ | ✅ |
| Weaker Arabian Sea (−3 m/s) | 96.4% | 28.2% outside | ✅ Day-10 max 0.64σ | ✅ |

### Qualification Gates

| Gate | Result |
|---|---|
| Valid / finite / non-negative `tp` | ✅ PASS |
| Beat Persistence at Day 4 | ✅ PASS |
| Causal shock sensible | ✅ PASS |
| 10-step latency < 5 s (CPU) | ✅ PASS — mean **580 ms** |

### Proxy Stage-1 Score: **45.17 / 70** automated points

| Component | Points | Max |
|---|---|---|
| Forecast Error | 11.00 | 30 |
| Heavy Rain CSI | 10.84 | 15 |
| Anti-Blur | 8.33 | 10 |
| Causal Shocks | **15.00** | 15 |

---

## ⚙️ Constraints & Limits

| Constraint | Requirement | Status |
|---|---|---|
| Offline inference | No internet, no weather APIs | ✅ |
| RAM | ≤ 6 GB | ✅ (~1.4 GB peak) |
| CPU cores | 4 cores | ✅ |
| GPU | None | ✅ CPU-only |
| Write path | Only `/tmp` (repo is read-only) | ✅ |
| Weight size | ≤ 2 GB | ✅ (~5.4 MB) |
| Latency | < 5 s per 10-step rollout | ✅ 580 ms mean |

**Known failure modes:**
- Day-8+ temperature (`t2m`) drift toward climatological mean — common for autoregressive models beyond predictability horizon (~7 days)
- Day-8 precipitation spatial variance ratio (`tp` = 0.29) is below the ideal [0.40, 2.50] band, indicating some heavy-rain smoothing at longer lead times
- `tcwv` can edge slightly negative in extreme edge cases before physical clamping corrects it

---

<div align="center">

**Track 2: Multiverse of Monsoon** · Yug Hackathon · Inspired by ISRO Bharatiya Antariksh Hackathon PS-5

</div>
