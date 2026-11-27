"""
benchmark_metrics.py — Hackathon Stage-1 & Stage-2 Evaluation Suite
====================================================================

Measures all Track 2 ("Multiverse of Monsoon") requirements on the locked
test split (2020-2021) or against a live inference server:

  1. Physical & Numerical Invariants (no NaN/Inf, tp >= 0, tcwv >= 0, bounds)
  2. Forecast Skill (Days 1, 4, 8, 10) vs Persistence & Train-Only Climatology:
       - Lat-weighted RMSE & Skill Score per channel
       - Qualification Gate: Beat Persistence at Day 4
       - Heavy-Rain Hits (>10 mm/day and >25 mm/day CSI, POD, FAR) & P99
  3. 5 Regime-Specific Windows (Monsoon Onset, Heavy Rain, Heatwave, Cyclone,
     Winter Disturbance) automatically mined from the 2020-2021 test set
  4. Dynamics / Anti-Blur (Days 1, 4, 8, 10):
       - Spatial Variance Ratio [0.40, 2.50]
       - Spatial Gradient (Sharpness) Ratio
  5. Causal Shocks (Zero Shock, Warm Bay of Bengal, Weaker Arabian Sea Wind):
       - Zero-shock exactness (< 1e-5 normalized diff)
       - Day-1 locality (energy fraction inside shock box + 1-cell halo)
       - Outward spread (outside energy fraction grows from Day 1 -> Day 4/8)
       - 10-day stability (finite, bounded response)
  6. CPU Inference Latency (< 5000 ms for 10-step rollout)

Usage:
    python tests/benchmark_metrics.py --checkpoint artifacts/primary_grast.pt
    python tests/benchmark_metrics.py --server http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CHANNELS,
    GRID_H,
    GRID_W,
    N_CHANNELS,
    DatasetReader,
    Transform,
    WindowIndex,
)
from src.data.torch_dataset import ClimateWindowDataset
from src.features.geo import build_geo_features
from src.models.grast import C_IN, GRAST, FallbackCNN
from src.rollout.engine import RolloutEngine, differentiable_rollout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lat_weights(lats: np.ndarray) -> np.ndarray:
    """Normalized cos(lat) weights summing to 1 over [32, 36]."""
    w = np.cos(np.deg2rad(lats)).astype("float64")
    w2d = np.broadcast_to(w[:, None], (GRID_H, GRID_W)).copy()
    w2d /= w2d.sum()
    return w2d


def build_seasonal_climatology(reader: DatasetReader) -> np.ndarray:
    """Train-only (2005-2017) day-of-year mean: [366, 6, 32, 36]."""
    train_idx = reader.split_indices("train")
    dates = reader._dates[train_idx]
    yr_starts = dates.astype("datetime64[Y]").astype("datetime64[D]")
    doys = (dates - yr_starts).astype(int)

    clim = np.zeros((366, N_CHANNELS, GRID_H, GRID_W), dtype="float64")
    counts = np.zeros(366, dtype="int32")
    for idx, d in zip(train_idx, doys):
        clim[d] += reader.data[idx]
        counts[d] += 1
    for d in range(366):
        clim[d] = clim[d] / counts[d] if counts[d] > 0 else clim[d - 1]
    return clim.astype("float32")


def spatial_grad_mag(arr: np.ndarray) -> np.ndarray:
    """Mean spatial gradient magnitude per sample & channel: [B, C]."""
    dy = np.diff(arr, axis=-2)
    dx = np.diff(arr, axis=-1)
    gy = np.mean(np.abs(dy), axis=(-2, -1))
    gx = np.mean(np.abs(dx), axis=(-2, -1))
    return 0.5 * (gy + gx)


def skill_score(rmse_model: float, rmse_ref: float) -> float:
    return 1.0 - (rmse_model / rmse_ref) if rmse_ref > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Full Locked-Test Rollout Evaluation (Skill, Rain CSI, Anti-Blur, Bounds)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full_test(
    reader: DatasetReader,
    transform: Transform,
    model: torch.nn.Module,
    geo_t: torch.Tensor,
    clim: np.ndarray,
    batch_size: int = 64,
    horizon: int = 10,
) -> dict:
    indices = reader.split_indices("test")
    win = WindowIndex(indices, n_inputs=3, max_horizon=horizon)
    ds = ClimateWindowDataset(reader, transform, win, horizon=horizon)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    std_t = torch.from_numpy(transform.std).float()
    mean_t = torch.from_numpy(transform.mean).float()
    w2d = lat_weights(reader.lats)

    leads = [1, 4, 8, 10]
    lead_idx = [l - 1 for l in leads]

    # [3 models (model, persist, clim), 4 leads, 6 channels]
    mse_acc = np.zeros((3, len(leads), N_CHANNELS), dtype="float64")
    # Heavy rain confusion matrices for >10 mm/day and >25 mm/day: [3, 4, 3 (hits, misses, fa)]
    rain10_cm = np.zeros((3, len(leads), 3), dtype="int64")
    rain25_cm = np.zeros((3, len(leads), 3), dtype="int64")

    # Anti-blur accumulators across leads [4, 6]
    var_pred = np.zeros((len(leads), N_CHANNELS), dtype="float64")
    var_true = np.zeros((len(leads), N_CHANNELS), dtype="float64")
    grad_pred = np.zeros((len(leads), N_CHANNELS), dtype="float64")
    grad_true = np.zeros((len(leads), N_CHANNELS), dtype="float64")

    rain_preds_d8, rain_trues_d8 = [], []

    # Physical validity tracking
    all_finite = True
    min_tp = float("inf")
    min_tcwv = float("inf")
    min_t2m = float("inf")
    max_t2m = float("-inf")
    min_msl = float("inf")
    max_msl = float("-inf")

    n_samples = 0
    win_ptr = 0
    model.eval()

    for x_norm, y_norm, doy_sin, doy_cos in loader:
        B = x_norm.shape[0]
        preds_norm = differentiable_rollout(
            model, transform, geo_t, x_norm, doy_sin, doy_cos,
            steps=horizon, std_t=std_t, mean_t=mean_t,
        )
        pred_phys = np.stack(
            [transform.denormalize(p.cpu().numpy()) for p in preds_norm], axis=1
        )
        # Match RolloutEngine physical clamping
        pred_phys[:, :, 1] = np.maximum(pred_phys[:, :, 1], 0.0)

        if not np.isfinite(pred_phys).all():
            all_finite = False
        min_tp = min(min_tp, float(pred_phys[:, :, 1].min()))
        min_tcwv = min(min_tcwv, float(pred_phys[:, :, 5].min()))
        min_t2m = min(min_t2m, float(pred_phys[:, :, 0].min()))
        max_t2m = max(max_t2m, float(pred_phys[:, :, 0].max()))
        min_msl = min(min_msl, float(pred_phys[:, :, 4].min()))
        max_msl = max(max_msl, float(pred_phys[:, :, 4].max()))

        true_phys = np.stack(
            [transform.denormalize(y_norm[:, k].numpy()) for k in range(horizon)], axis=1
        )
        last_phys = transform.denormalize(x_norm[:, -1].numpy())
        pers_phys = np.broadcast_to(last_phys[:, None], (B, horizon, N_CHANNELS, GRID_H, GRID_W))

        clim_batch = np.empty((B, horizon, N_CHANNELS, GRID_H, GRID_W), dtype="float32")
        for b in range(B):
            _, tgt_days = win[win_ptr + b]
            dates = reader._dates[tgt_days[:horizon]]
            yr_starts = dates.astype("datetime64[Y]").astype("datetime64[D]")
            doys = np.clip((dates - yr_starts).astype(int), 0, 365)
            clim_batch[b] = clim[doys]
        win_ptr += B

        candidates = [pred_phys, pers_phys, clim_batch]

        for li, k in enumerate(lead_idx):
            t_k = true_phys[:, k]
            p_k = pred_phys[:, k]

            var_pred[li] += p_k.var(axis=(2, 3)).sum(axis=0)
            var_true[li] += t_k.var(axis=(2, 3)).sum(axis=0)
            grad_pred[li] += spatial_grad_mag(p_k).sum(axis=0)
            grad_true[li] += spatial_grad_mag(t_k).sum(axis=0)

            for mi, cand in enumerate(candidates):
                c_k = cand[:, k]
                wmse = (((c_k - t_k) ** 2) * w2d[None, None, :, :]).sum(axis=(2, 3)).sum(axis=0)
                mse_acc[mi, li] += wmse

                for cm, thr in ((rain10_cm, 10.0), (rain25_cm, 25.0)):
                    pr = c_k[:, 1] > thr
                    tr = t_k[:, 1] > thr
                    cm[mi, li, 0] += int((pr & tr).sum())
                    cm[mi, li, 1] += int((~pr & tr).sum())
                    cm[mi, li, 2] += int((pr & ~tr).sum())

        rain_preds_d8.append(pred_phys[:, 7, 1].ravel())
        rain_trues_d8.append(true_phys[:, 7, 1].ravel())
        n_samples += B

    rmse = np.sqrt(mse_acc / n_samples)

    leads_out = {}
    for li, lead in enumerate(leads):
        ld = {}
        skills_best, skills_pers = [], []
        for ci, ch in enumerate(CHANNELS):
            rm = float(rmse[0, li, ci])
            rp = float(rmse[1, li, ci])
            rc = float(rmse[2, li, ci])
            best_ref = min(rp, rc)
            sb = skill_score(rm, best_ref)
            sp = skill_score(rm, rp)
            skills_best.append(sb)
            skills_pers.append(sp)
            ld[ch] = {
                "rmse_model": round(rm, 4),
                "rmse_persist": round(rp, 4),
                "rmse_clim": round(rc, 4),
                "skill_vs_best_ref": round(sb, 4),
                "skill_vs_persist": round(sp, 4),
            }
        ld["mean_skill_vs_best_ref"] = round(float(np.mean(skills_best)), 4)
        ld["mean_skill_vs_persist"] = round(float(np.mean(skills_pers)), 4)

        for tag, cm in (("rain_gt10mm", rain10_cm), ("rain_gt25mm", rain25_cm)):
            rd = {}
            for mi, mname in enumerate(("model", "persist", "clim")):
                h, m, fa = cm[mi, li]
                csi = h / (h + m + fa) if (h + m + fa) > 0 else 0.0
                pod = h / (h + m) if (h + m) > 0 else 0.0
                prec = h / (h + fa) if (h + fa) > 0 else 0.0
                far = fa / (h + fa) if (h + fa) > 0 else 0.0
                rd[mname] = {
                    "csi": round(float(csi), 4),
                    "pod_recall": round(float(pod), 4),
                    "precision": round(float(prec), 4),
                    "far": round(float(far), 4),
                }
            ld[tag] = rd

        vr = var_pred[li] / np.maximum(var_true[li], 1e-12)
        gr = grad_pred[li] / np.maximum(grad_true[li], 1e-12)
        ld["spatial_var_ratio"] = {ch: round(float(vr[ci]), 4) for ci, ch in enumerate(CHANNELS)}
        ld["spatial_grad_ratio"] = {ch: round(float(gr[ci]), 4) for ci, ch in enumerate(CHANNELS)}
        leads_out[f"d{lead}"] = ld

    p_all = np.concatenate(rain_preds_d8)
    t_all = np.concatenate(rain_trues_d8)

    return {
        "n_windows": n_samples,
        "physical_validity": {
            "all_finite": all_finite,
            "rain_non_negative": min_tp >= 0.0,
            "min_tp_mm": round(min_tp, 5),
            "min_tcwv": round(min_tcwv, 3),
            "t2m_range_K": [round(min_t2m, 2), round(max_t2m, 2)],
            "msl_range_Pa": [round(min_msl, 1), round(max_msl, 1)],
        },
        "leads": leads_out,
        "day8_rain_p99": {
            "pred_mm": round(float(np.percentile(p_all, 99)), 3),
            "true_mm": round(float(np.percentile(t_all, 99)), 3),
        },
    }


# ---------------------------------------------------------------------------
# 2. 5 Regime-Specific Weather Windows (from Locked Test 2020-2021)
# ---------------------------------------------------------------------------

def select_regime_windows(reader: DatasetReader) -> dict[str, int]:
    """Find 5 representative window indices in the test split (2020-2021)."""
    test_idx = reader.split_indices("test")
    win = WindowIndex(test_idx, n_inputs=3, max_horizon=10)
    dates = reader._dates
    land_mask = reader.static[0] > 0.5

    best = {
        "monsoon_onset": (-1, -1e9),
        "heavy_rain": (-1, -1e9),
        "heatwave": (-1, -1e9),
        "cyclone": (-1, 1e9),
        "winter_disturbance": (-1, -1e9),
    }

    for wi in range(len(win)):
        inp_days, tgt_days = win[wi]
        d0 = dates[tgt_days[0]]
        month = int(str(d0)[5:7])
        tgt = reader.data[tgt_days[:10]]  # [10, 6, 32, 36]

        # 1. Monsoon onset (early June): strong rain ramp-up + SW wind
        if month == 6:
            day_num = int(str(d0)[8:10])
            if day_num <= 15:
                score = float(tgt[:, 1].mean() + 0.5 * tgt[:, 2].mean())
                if score > best["monsoon_onset"][1]:
                    best["monsoon_onset"] = (wi, score)

        # 2. Heavy rain (JJAS peak precipitation)
        if month in (6, 7, 8, 9):
            score = float(np.percentile(tgt[:, 1], 99))
            if score > best["heavy_rain"][1]:
                best["heavy_rain"] = (wi, score)

        # 3. Heatwave (Apr-May peak land t2m)
        if month in (4, 5):
            score = float(tgt[:, 0][:, land_mask].max())
            if score > best["heatwave"][1]:
                best["heatwave"] = (wi, score)

        # 4. Cyclone / deep low (minimum MSL over domain)
        min_p = float(tgt[:, 4].min())
        if min_p < best["cyclone"][1]:
            best["cyclone"] = (wi, min_p)

        # 5. Winter disturbance (Dec-Feb northern/western wind+rain variance)
        if month in (12, 1, 2):
            score = float(tgt[:, 1, 20:, :].mean() + tgt[:, 2, 20:, :].std())
            if score > best["winter_disturbance"][1]:
                best["winter_disturbance"] = (wi, score)

    return {k: v[0] for k, v in best.items()}


def evaluate_regime_windows(
    rollout_fn: Callable[[np.ndarray, np.ndarray, int, Optional[np.ndarray]], np.ndarray],
    reader: DatasetReader,
    transform: Transform,
    clim: np.ndarray,
) -> dict:
    test_idx = reader.split_indices("test")
    win = WindowIndex(test_idx, n_inputs=3, max_horizon=10)
    w2d = lat_weights(reader.lats)
    regimes = select_regime_windows(reader)

    out = {}
    for name, wi in regimes.items():
        inp_days, tgt_days = win[wi]
        init_states = reader.data[inp_days].copy()
        doy = reader.doy_sincos(inp_days[-1])
        true_phys = reader.data[tgt_days[:10]].copy()

        pred_phys = rollout_fn(init_states, doy, 10, None)
        pers_phys = np.broadcast_to(init_states[-1][None], (10, N_CHANNELS, GRID_H, GRID_W))

        dates = reader._dates[tgt_days[:10]]
        yr_starts = dates.astype("datetime64[Y]").astype("datetime64[D]")
        doys = np.clip((dates - yr_starts).astype(int), 0, 365)
        clim_phys = clim[doys]

        # Normalized weighted RMSE across all 6 channels at days 1, 4, 8
        regime_res = {"start_date": str(dates[0])}
        for lead in (1, 4, 8):
            k = lead - 1
            rm = float(np.sqrt((((pred_phys[k] - true_phys[k]) / transform.std[:, None, None]) ** 2 * w2d[None]).sum(axis=(1, 2)).mean()))
            rp = float(np.sqrt((((pers_phys[k] - true_phys[k]) / transform.std[:, None, None]) ** 2 * w2d[None]).sum(axis=(1, 2)).mean()))
            rc = float(np.sqrt((((clim_phys[k] - true_phys[k]) / transform.std[:, None, None]) ** 2 * w2d[None]).sum(axis=(1, 2)).mean()))
            best_ref = min(rp, rc)
            regime_res[f"d{lead}_norm_skill_vs_best"] = round(skill_score(rm, best_ref), 4)
            regime_res[f"d{lead}_norm_skill_vs_persist"] = round(skill_score(rm, rp), 4)
        out[name] = regime_res
    return out


# ---------------------------------------------------------------------------
# 3. Causal Shock Tests (Zero, Warm Bay of Bengal, Weaker Arabian Sea Wind)
# ---------------------------------------------------------------------------

def evaluate_shocks_and_latency(
    rollout_fn: Callable[[np.ndarray, np.ndarray, int, Optional[np.ndarray]], np.ndarray],
    reader: DatasetReader,
    transform: Transform,
) -> dict:
    test_idx = reader.split_indices("test")
    win = WindowIndex(test_idx, n_inputs=3, max_horizon=10)

    # Pick a monsoon-season window (June/July) for physically meaningful shocks
    shock_wi = 0
    for wi in range(len(win)):
        _, tgt_days = win[wi]
        if str(reader._dates[tgt_days[0]])[5:7] == "07":
            shock_wi = wi
            break

    inp_days, _ = win[shock_wi]
    init_states = reader.data[inp_days].copy()
    doy = reader.doy_sincos(inp_days[-1])

    prime = rollout_fn(init_states, doy, 10, None)

    # 1. Zero shock
    zero_pert = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype="float32")
    zero_out = rollout_fn(init_states, doy, 10, zero_pert)
    zero_max_norm_diff = float(np.max(np.abs(prime - zero_out) / transform.std[None, :, None, None]))

    # Lat/Lon masks for Bay of Bengal & Arabian Sea
    lats = reader.lats[:, None]   # [32, 1]
    lons = reader.lons[None, :]   # [1, 36]
    ocean = reader.static[0] < 0.5

    bob_box = (lats >= 8.5) & (lats <= 20.5) & (lons >= 82.5) & (lons <= 94.5) & ocean
    bob_halo = (lats >= 7.5) & (lats <= 21.5) & (lons >= 81.5) & (lons <= 95.5)

    arab_box = (lats >= 8.5) & (lats <= 20.5) & (lons >= 65.5) & (lons <= 75.5) & ocean
    arab_halo = (lats >= 7.5) & (lats <= 21.5) & (lons >= 64.5) & (lons <= 76.5)

    shock_specs = [
        ("warm_bay_of_bengal", 0, +2.0, bob_box, bob_halo),
        ("weaker_arabian_sea_wind", 2, -3.0, arab_box, arab_halo),
    ]

    shocks = {}
    for label, ch_idx, mag, box_mask, halo_mask in shock_specs:
        pert = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype="float32")
        pert[ch_idx][box_mask] = mag
        multi = rollout_fn(init_states, doy, 10, pert)
        diff = multi - prime  # [10, 6, 32, 36]

        finite = bool(np.isfinite(diff).all())
        rain_nonneg = bool((multi[:, 1] >= 0.0).all())
        energy = (diff / transform.std[None, :, None, None]) ** 2  # [10, 6, 32, 36]

        def outside_frac(step_idx: int) -> float:
            tot = float(energy[step_idx].sum())
            if tot <= 1e-12:
                return 0.0
            inside = float(energy[step_idx][:, halo_mask].sum())
            return (tot - inside) / tot

        d1_out = outside_frac(0)
        d4_out = outside_frac(3)
        d8_out = outside_frac(7)
        d1_local_frac = 1.0 - d1_out
        d10_max_norm = float(np.max(np.abs(diff[9]) / transform.std[:, None, None]))
        d1_max_norm = float(np.max(np.abs(diff[0]) / transform.std[:, None, None]))

        # Causal pass: local at Day 1 (>= 0.50), spreads outward (d4 or d8 > d1), finite & bounded (< 5.0 std)
        causal_pass = (
            finite
            and rain_nonneg
            and d1_max_norm > 1e-3
            and d1_local_frac >= 0.50
            and max(d4_out, d8_out) > d1_out + 0.01
            and d10_max_norm < 5.0
        )

        shocks[label] = {
            "finite": finite,
            "rain_non_negative": rain_nonneg,
            "d1_local_energy_frac": round(d1_local_frac, 4),
            "d1_outside_frac": round(d1_out, 4),
            "d4_outside_frac": round(d4_out, 4),
            "d8_outside_frac": round(d8_out, 4),
            "d1_max_norm_resp": round(d1_max_norm, 4),
            "d10_max_norm_resp": round(d10_max_norm, 4),
            "causal_pass": causal_pass,
        }

    # Latency (10 calls of 10-step rollout)
    latencies_ms = []
    for i in range(10):
        inp_i, _ = win[i * 10]
        st = reader.data[inp_i]
        dy = reader.doy_sincos(inp_i[-1])
        t0 = time.perf_counter()
        _ = rollout_fn(st, dy, 10, None)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "zero_shock_max_norm_diff": zero_max_norm_diff,
        "zero_shock_pass": zero_max_norm_diff < 1e-5,
        "shocks": shocks,
        "latency_10step_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2),
            "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            "max": round(float(np.max(latencies_ms)), 2),
            "under_5000ms": float(np.max(latencies_ms)) < 5000.0,
        },
    }


# ---------------------------------------------------------------------------
# 4. Qualification Gates & Composite Stage-1 Readiness Score
# ---------------------------------------------------------------------------

def compute_scorecard(test_res: dict, shock_res: dict) -> dict:
    leads = test_res["leads"]
    pv = test_res["physical_validity"]

    # Qualification gates
    gate_valid = pv["all_finite"] and pv["rain_non_negative"]
    gate_d4_persist = leads["d4"]["mean_skill_vs_persist"] > 0.0
    gate_shock = shock_res["zero_shock_pass"] and any(
        s["causal_pass"] for s in shock_res["shocks"].values()
    )
    gate_latency = shock_res["latency_10step_ms"]["under_5000ms"]

    # Proxy Stage-1 automated score (out of 70 automated technical points: 45 skill + 25 dynamics)
    # 1. Error Skill (30 pts): mean skill vs best_ref across d1, d4, d8 (capped [0, 0.35] -> 0..30)
    mean_err_skill = np.mean([leads[d]["mean_skill_vs_best_ref"] for d in ("d1", "d4", "d8")])
    pts_error = float(np.clip(mean_err_skill / 0.30, 0.0, 1.0) * 30.0)

    # 2. Heavy-Rain Hits (15 pts): mean CSI (>10mm) ratio vs best_ref CSI across d1, d4, d8
    csi_scores = []
    for d in ("d1", "d4", "d8"):
        rm = leads[d]["rain_gt10mm"]["model"]["csi"]
        rb = max(leads[d]["rain_gt10mm"]["persist"]["csi"], leads[d]["rain_gt10mm"]["clim"]["csi"], 1e-4)
        csi_scores.append(min(rm / rb, 1.5) / 1.5)
    pts_rain = float(np.mean(csi_scores) * 15.0)

    # 3. Anti-Blur (10 pts): Day-8 spatial variance ratio in [0.40, 2.50] across 6 channels
    d8_vr = list(leads["d8"]["spatial_var_ratio"].values())
    in_band = sum(1 for v in d8_vr if 0.40 <= v <= 2.50) / len(d8_vr)
    pts_noblur = float(in_band * 10.0)

    # 4. Causal Shocks (15 pts): 5 pts zero shock + 5 pts BoB + 5 pts Arabian Sea
    pts_shocks = (5.0 if shock_res["zero_shock_pass"] else 0.0) + sum(
        5.0 if s["causal_pass"] else 0.0 for s in shock_res["shocks"].values()
    )

    return {
        "qualification_gates": {
            "valid_finite_nonneg": gate_valid,
            "beat_persist_day4": gate_d4_persist,
            "causal_shock_ok": gate_shock,
            "latency_under_5s": gate_latency,
            "ALL_GATES_PASS": gate_valid and gate_d4_persist and gate_shock and gate_latency,
        },
        "proxy_stage1_points": {
            "forecast_error_30": round(pts_error, 2),
            "heavy_rain_15": round(pts_rain, 2),
            "anti_blur_10": round(pts_noblur, 2),
            "causal_shocks_15": round(pts_shocks, 2),
            "total_auto_70": round(pts_error + pts_rain + pts_noblur + pts_shocks, 2),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="artifacts/primary_grast.pt")
    ap.add_argument("--server", type=str, default=None, help="Optional live serve.py URL for shock/regime/latency checks")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=str, default="artifacts/benchmark_report.json")
    args = ap.parse_args()

    reader = DatasetReader()
    transform = Transform()
    clim = build_seasonal_climatology(reader)

    if args.server:
        import urllib.request

        base = args.server.rstrip("/")
        with urllib.request.urlopen(f"{base}/health", timeout=10) as resp:
            health = json.loads(resp.read().decode())
        print(f"Connected to server: {health}")

        def server_rollout(init_states: np.ndarray, doy: np.ndarray, steps: int, pert: Optional[np.ndarray]) -> np.ndarray:
            payload = json.dumps({
                "init_states": init_states.tolist(),
                "doy_sincos": [float(doy[0]), float(doy[1])],
                "steps": int(steps),
                "perturbation": None if pert is None else pert.tolist(),
            }).encode()
            req = urllib.request.Request(f"{base}/simulate", data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return np.asarray(data["rollout"], dtype="float32")

        print("\n=== Evaluating 5 Regime Windows (via Server) ===")
        regime_res = evaluate_regime_windows(server_rollout, reader, transform, clim)
        print("\n=== Evaluating Causal Shocks & Latency (via Server) ===")
        shock_res = evaluate_shocks_and_latency(server_rollout, reader, transform)
        report = {"server": base, "health": health, "regimes": regime_res, "shocks_and_latency": shock_res}
        out_path = ROOT / args.out
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    # Local checkpoint evaluation
    ckpt_path = ROOT / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_ch = ckpt.get("base_ch", 64)
    if ckpt.get("model_class", "primary") == "primary":
        model = GRAST(c_in=ckpt.get("c_in", C_IN), base_ch=base_ch)
    else:
        model = FallbackCNN(c_in=ckpt.get("c_in", C_IN), base_ch=base_ch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    geo_np = build_geo_features(reader.static[0], reader.static[1], reader.lats, reader.lons)
    geo_t = torch.from_numpy(geo_np).unsqueeze(0).float()
    engine = RolloutEngine(model, transform, geo_np, torch.device("cpu"))

    print(f"Loaded {ckpt_path.name} ({ckpt.get('model_name')}, params={ckpt.get('n_params')})")
    print("\n=== 1. Locked Test (2020-2021) Full Evaluation ===")
    test_res = evaluate_full_test(reader, transform, model, geo_t, clim, batch_size=args.batch)

    print("=== 2. 5 Regime-Specific Windows ===")
    regime_res = evaluate_regime_windows(engine.rollout, reader, transform, clim)

    print("=== 3. Causal Shocks & CPU Latency ===")
    shock_res = evaluate_shocks_and_latency(engine.rollout, reader, transform)

    scorecard = compute_scorecard(test_res, shock_res)
    report = {
        "checkpoint": str(args.checkpoint),
        "model_name": ckpt.get("model_name"),
        "n_params": ckpt.get("n_params"),
        "scorecard": scorecard,
        "test": test_res,
        "regimes": regime_res,
        "shocks_and_latency": shock_res,
    }

    out_path = ROOT / args.out
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary printout
    print("\n" + "=" * 84)
    print("STAGE-1 BENCHMARK SUMMARY (Locked Test 2020-2021)")
    print("=" * 84)
    hdr = f"{'Lead':4s} | " + " | ".join(f"{c:>7s}" for c in CHANNELS) + " | MeanRef | CSI>10 | CSI>25"
    print(hdr)
    print("-" * len(hdr))
    for d in ("d1", "d4", "d8", "d10"):
        ld = test_res["leads"][d]
        vals = " | ".join(f"{ld[c]['skill_vs_best_ref']:+7.3f}" for c in CHANNELS)
        ms = ld["mean_skill_vs_best_ref"]
        c10 = ld["rain_gt10mm"]["model"]["csi"]
        c25 = ld["rain_gt25mm"]["model"]["csi"]
        print(f"{d:4s} | {vals} | {ms:+7.3f} | {c10:6.3f} | {c25:6.3f}")

    print("\nDay-8 Spatial Variance Ratio (gate [0.40, 2.50]):", test_res["leads"]["d8"]["spatial_var_ratio"])
    print("Day-8 Rain P99 (mm/day):", test_res["day8_rain_p99"])
    print("\nQualification Gates:", json.dumps(scorecard["qualification_gates"], indent=2))
    print("Proxy Stage-1 Score (out of 70 auto pts):", json.dumps(scorecard["proxy_stage1_points"], indent=2))
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
