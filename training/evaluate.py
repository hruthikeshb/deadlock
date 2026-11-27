"""
evaluate.py — Comprehensive evaluation & diagnostic suite for trained models.
=============================================================================

Evaluates:
  1. Training health (training_log.jsonl & checkpoint integrity)
  2. Mandatory baselines (Persistence & train-only Seasonal Climatology)
  3. 10-day autoregressive rollout on Val (2018-2019) and Locked Test (2020-2021):
       - Latitude-weighted RMSE & Skill Score at Days 1, 4, 8, 10
       - Heavy-rain (>10 mm/day) CSI, Precision, Recall, and P99
       - Day-8 Spatial Variance Ratio (anti-blur gate [0.40, 2.50])
  4. Perturbation sanity (null consistency + localized shock propagation)
  5. 10-step CPU inference latency via RolloutEngine

Usage:
    python training/evaluate.py --checkpoint artifacts/primary_grast.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CHANNELS, DatasetReader, Transform, WindowIndex, N_CHANNELS, GRID_H, GRID_W
from src.data.torch_dataset import ClimateWindowDataset
from src.features.geo import build_geo_features
from src.models.grast import GRAST, FallbackCNN, C_IN
from src.rollout.engine import RolloutEngine, differentiable_rollout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="artifacts/primary_grast.pt")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", type=str, default="artifacts/eval_report.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Training Log & Checkpoint Audit
# ---------------------------------------------------------------------------

def audit_training(ckpt_path: Path, log_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    all_finite = all(torch.isfinite(v).all().item() for v in state.values())
    res_scale = state["residual_scale"].numpy().tolist() if "residual_scale" in state else None

    log_summary = {}
    if log_path.exists():
        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        for phase in ("h1", "h2", "h4"):
            pe = [e for e in entries if e.get("phase") == phase]
            if pe:
                log_summary[phase] = {
                    "count": len(pe),
                    "min_train_loss": min(e["train_loss"] for e in pe),
                    "min_val_loss": min(e["val_loss"] for e in pe),
                    "last_val_loss": pe[-1]["val_loss"],
                }

    return {
        "model_name": ckpt.get("model_name"),
        "model_class": ckpt.get("model_class"),
        "base_ch": ckpt.get("base_ch", 64),
        "n_params": ckpt.get("n_params"),
        "trained_horizon": ckpt.get("horizon"),
        "all_weights_finite": all_finite,
        "residual_scale": res_scale,
        "best_ckpt_val_loss": ckpt.get("best_metrics", {}).get("val_loss"),
        "log_phases": log_summary,
    }


# ---------------------------------------------------------------------------
# 2. Train-only Seasonal Climatology
# ---------------------------------------------------------------------------

def build_seasonal_climatology(reader: DatasetReader) -> np.ndarray:
    """
    Compute train-only (2005-2017) day-of-year mean: shape [366, 6, 32, 36].
    """
    train_idx = reader.split_indices("train")
    dates = reader._dates[train_idx]
    yr_starts = dates.astype("datetime64[Y]").astype("datetime64[D]")
    doys = (dates - yr_starts).astype(int)  # 0-based (0..365)

    clim = np.zeros((366, N_CHANNELS, GRID_H, GRID_W), dtype="float64")
    counts = np.zeros(366, dtype="int32")

    for idx, d in zip(train_idx, doys):
        clim[d] += reader.data[idx]
        counts[d] += 1

    for d in range(366):
        if counts[d] > 0:
            clim[d] /= counts[d]
        else:
            clim[d] = clim[d - 1]
    return clim.astype("float32")


# ---------------------------------------------------------------------------
# 3. Split Evaluation (Persistence, Climatology, Model)
# ---------------------------------------------------------------------------

def lat_weights(lats: np.ndarray) -> np.ndarray:
    """Normalized cos(lat) weights summing to 1 over [H, W]."""
    w = np.cos(np.deg2rad(lats)).astype("float64")
    w2d = np.broadcast_to(w[:, None], (GRID_H, GRID_W)).copy()
    w2d /= w2d.sum()
    return w2d  # [32, 36]


@torch.no_grad()
def evaluate_split(
    split_name: str,
    reader: DatasetReader,
    transform: Transform,
    model: torch.nn.Module,
    geo_t: torch.Tensor,
    clim: np.ndarray,
    batch_size: int = 64,
    horizon: int = 10,
) -> dict:
    indices = reader.split_indices(split_name)
    win = WindowIndex(indices, n_inputs=3, max_horizon=horizon)
    ds = ClimateWindowDataset(reader, transform, win, horizon=horizon)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    std_t = torch.from_numpy(transform.std).float().to(device)
    mean_t = torch.from_numpy(transform.mean).float().to(device)
    w2d = lat_weights(reader.lats)  # [32, 36]

    leads = [1, 4, 8, 10]
    lead_idx = [l - 1 for l in leads]

    # Accumulators for weighted MSE: [3 models, 4 leads, 6 channels]
    # models: 0=model, 1=persistence, 2=climatology
    mse_acc = np.zeros((3, len(leads), N_CHANNELS), dtype="float64")
    n_samples = 0

    # Rain CSI (>10 mm/day) accumulators: [3 models, 4 leads, 3 (hits, misses, false_alarms)]
    rain_cm = np.zeros((3, len(leads), 3), dtype="int64")
    rain_preds_d8 = []
    rain_trues_d8 = []

    # Day-8 spatial variance ratio accumulators (store per-sample spatial var or full grids)
    # Using mean spatial variance across samples: E[Var_HW(pred)] / E[Var_HW(true)]
    var_pred_d8 = np.zeros(N_CHANNELS, dtype="float64")
    var_true_d8 = np.zeros(N_CHANNELS, dtype="float64")

    # Precompute target doy indices for all windows in order
    win_ptr = 0
    model.eval()

    for x_norm, y_norm, doy_sin, doy_cos in loader:
        B = x_norm.shape[0]
        preds_norm = differentiable_rollout(
            model, transform, geo_t, x_norm, doy_sin, doy_cos,
            steps=horizon, std_t=std_t, mean_t=mean_t,
        )
        # Stack to [B, horizon, 6, 32, 36] in physical units
        pred_phys = np.stack(
            [transform.denormalize(p.cpu().numpy()) for p in preds_norm],
            axis=1,
        )
        pred_phys[:, :, 1] = np.maximum(pred_phys[:, :, 1], 0.0)

        # True physical targets [B, horizon, 6, 32, 36]
        true_phys = np.stack(
            [transform.denormalize(y_norm[:, k].numpy()) for k in range(horizon)],
            axis=1,
        )

        # Persistence: repeat last input day [B, 6, 32, 36]
        last_phys = transform.denormalize(x_norm[:, -1].numpy())  # [B, 6, 32, 36]
        pers_phys = np.broadcast_to(last_phys[:, None], (B, horizon, N_CHANNELS, GRID_H, GRID_W))

        # Climatology: lookup target dates
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
            t_k = true_phys[:, k]  # [B, 6, 32, 36]
            for mi, cand in enumerate(candidates):
                c_k = cand[:, k]
                diff2 = (c_k - t_k) ** 2  # [B, 6, 32, 36]
                # weighted mean over H, W, summed over B
                wmse = (diff2 * w2d[None, None, :, :]).sum(axis=(2, 3)).sum(axis=0)  # [6]
                mse_acc[mi, li] += wmse

                # Rain CSI > 10 mm/day
                p_rain = c_k[:, 1] > 10.0
                t_rain = t_k[:, 1] > 10.0
                hits = int((p_rain & t_rain).sum())
                misses = int((~p_rain & t_rain).sum())
                fa = int((p_rain & ~t_rain).sum())
                rain_cm[mi, li, 0] += hits
                rain_cm[mi, li, 1] += misses
                rain_cm[mi, li, 2] += fa

        # Day-8 spatial variance (k=7)
        var_pred_d8 += pred_phys[:, 7].var(axis=(2, 3)).sum(axis=0)
        var_true_d8 += true_phys[:, 7].var(axis=(2, 3)).sum(axis=0)
        rain_preds_d8.append(pred_phys[:, 7, 1].ravel())
        rain_trues_d8.append(true_phys[:, 7, 1].ravel())

        n_samples += B

    rmse = np.sqrt(mse_acc / n_samples)  # [3, 4, 6]

    out = {"n_windows": n_samples, "leads": {}}
    for li, lead in enumerate(leads):
        lead_dict = {}
        for ci, ch in enumerate(CHANNELS):
            r_mod = float(rmse[0, li, ci])
            r_per = float(rmse[1, li, ci])
            r_clm = float(rmse[2, li, ci])
            best_ref = min(r_per, r_clm)
            skill_ref = 1.0 - r_mod / best_ref if best_ref > 0 else 0.0
            skill_per = 1.0 - r_mod / r_per if r_per > 0 else 0.0
            lead_dict[ch] = {
                "rmse_model": round(r_mod, 4),
                "rmse_persist": round(r_per, 4),
                "rmse_clim": round(r_clm, 4),
                "skill_vs_best_ref": round(skill_ref, 4),
                "skill_vs_persist": round(skill_per, 4),
            }

        # Rain CSI (>10 mm/day)
        csi_dict = {}
        for mi, mname in enumerate(("model", "persist", "clim")):
            h, m, fa = rain_cm[mi, li]
            csi = h / (h + m + fa) if (h + m + fa) > 0 else 0.0
            prec = h / (h + fa) if (h + fa) > 0 else 0.0
            rec = h / (h + m) if (h + m) > 0 else 0.0
            csi_dict[mname] = {
                "csi": round(float(csi), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
            }
        lead_dict["rain_gt10mm"] = csi_dict
        out["leads"][f"d{lead}"] = lead_dict

    # Day-8 spatial variance ratio & rain P99
    var_ratio = var_pred_d8 / np.maximum(var_true_d8, 1e-12)
    out["day8_spatial_var_ratio"] = {
        ch: round(float(var_ratio[ci]), 4) for ci, ch in enumerate(CHANNELS)
    }
    p_all = np.concatenate(rain_preds_d8)
    t_all = np.concatenate(rain_trues_d8)
    out["day8_rain_p99"] = {
        "pred_mm": round(float(np.percentile(p_all, 99)), 3),
        "true_mm": round(float(np.percentile(t_all, 99)), 3),
        "pred_mean_mm": round(float(p_all.mean()), 3),
        "true_mean_mm": round(float(t_all.mean()), 3),
    }
    return out


# ---------------------------------------------------------------------------
# 4. Perturbation & Runtime Checks
# ---------------------------------------------------------------------------

def evaluate_perturbations_and_latency(
    engine: RolloutEngine,
    reader: DatasetReader,
    transform: Transform,
) -> dict:
    test_idx = reader.split_indices("test")
    win = WindowIndex(test_idx, n_inputs=3, max_horizon=10)
    inp_days, _ = win[0]
    init_states = reader.data[inp_days].copy()
    doy = reader.doy_sincos(inp_days[-1])

    # Null consistency
    prime = engine.rollout(init_states, doy, steps=10, perturbation=None)
    zero_pert = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype="float32")
    zero_out = engine.rollout(init_states, doy, steps=10, perturbation=zero_pert)
    max_null_diff_norm = float(
        np.max(np.abs(prime - zero_out) / transform.std[None, :, None, None])
    )

    # Localized shocks at center (16, 18)
    shocks = {}
    for label, ch_idx, mag in [("t2m_+2K", 0, 2.0), ("tp_+10mm", 1, 10.0)]:
        pert = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype="float32")
        pert[ch_idx, 16, 18] = mag
        multi = engine.rollout(init_states, doy, steps=10, perturbation=pert)
        diff = multi - prime  # [10, 6, 32, 36]
        finite = bool(np.isfinite(diff).all())

        # Day-1 energy in 5x5 neighborhood around (16, 18) vs total domain
        d1_e = (diff[0] / transform.std[:, None, None]) ** 2
        total_e = float(d1_e.sum())
        near_e = float(d1_e[:, 14:19, 16:21].sum())
        near_frac = near_e / total_e if total_e > 0 else 0.0

        # Max normalized response at day 1 and day 10
        d1_max = float(np.max(np.abs(diff[0]) / transform.std[:, None, None]))
        d10_max = float(np.max(np.abs(diff[9]) / transform.std[:, None, None]))

        shocks[label] = {
            "finite": finite,
            "d1_near5x5_energy_frac": round(near_frac, 4),
            "d1_max_norm_resp": round(d1_max, 4),
            "d10_max_norm_resp": round(d10_max, 4),
        }

    # Latency (10 runs of 10-step rollout)
    latencies_ms = []
    for i in range(10):
        inp_i, _ = win[i * 10]
        st = reader.data[inp_i]
        dy = reader.doy_sincos(inp_i[-1])
        t0 = time.perf_counter()
        _ = engine.rollout(st, dy, steps=10, perturbation=None)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "null_max_norm_diff": max_null_diff_norm,
        "null_pass_1e_3": max_null_diff_norm < 1e-3,
        "shocks": shocks,
        "latency_10step_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2),
            "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            "max": round(float(np.max(latencies_ms)), 2),
            "under_5000ms": float(np.max(latencies_ms)) < 5000.0,
        },
    }


def main() -> None:
    args = parse_args()
    ckpt_path = ROOT / args.checkpoint
    log_path = ROOT / "artifacts" / "training_log.jsonl"

    print("=== 1. Training Audit ===")
    audit = audit_training(ckpt_path, log_path)
    print(json.dumps(audit, indent=2))

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_ch = ckpt.get("base_ch", 64)
    if ckpt.get("model_class", "primary") == "primary":
        model = GRAST(c_in=ckpt.get("c_in", C_IN), base_ch=base_ch)
    else:
        model = FallbackCNN(c_in=ckpt.get("c_in", C_IN), base_ch=base_ch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    reader = DatasetReader()
    transform = Transform()
    geo_np = build_geo_features(reader.static[0], reader.static[1], reader.lats, reader.lons)
    geo_t = torch.from_numpy(geo_np).unsqueeze(0).float()

    print("\n=== 2. Building Train-Only Climatology ===")
    clim = build_seasonal_climatology(reader)

    print("\n=== 3. Evaluating Validation (2018-2019) ===")
    val_res = evaluate_split("val", reader, transform, model, geo_t, clim, batch_size=args.batch)

    print("\n=== 4. Evaluating Locked Test (2020-2021) ===")
    test_res = evaluate_split("test", reader, transform, model, geo_t, clim, batch_size=args.batch)

    print("\n=== 5. Perturbation & CPU Latency Checks ===")
    engine = RolloutEngine(model, transform, geo_np, torch.device("cpu"))
    pert_res = evaluate_perturbations_and_latency(engine, reader, transform)

    report = {
        "audit": audit,
        "val": val_res,
        "test": test_res,
        "perturbation_and_latency": pert_res,
    }

    out_path = ROOT / args.out
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print summary table for Test
    print("\n" + "=" * 78)
    print("LOCKED TEST (2020-2021) SUMMARY — Skill vs Best Ref (Persist / Clim)")
    print("=" * 78)
    hdr = f"{'Lead':5s} | " + " | ".join(f"{c:>8s}" for c in CHANNELS) + " | RainCSI"
    print(hdr)
    print("-" * len(hdr))
    for d in ("d1", "d4", "d8", "d10"):
        ld = test_res["leads"][d]
        vals = " | ".join(f"{ld[c]['skill_vs_best_ref']:+8.3f}" for c in CHANNELS)
        csi = ld["rain_gt10mm"]["model"]["csi"]
        print(f"{d:5s} | {vals} | {csi:6.3f}")

    print("\nDay-8 Spatial Variance Ratio (gate [0.40, 2.50]):", test_res["day8_spatial_var_ratio"])
    print("Day-8 Rain P99:", test_res["day8_rain_p99"])
    print("Perturbation & Latency:", json.dumps(pert_res, indent=2))
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
