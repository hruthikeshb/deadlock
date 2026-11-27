"""
train.py — Main training entrypoint for GRAST and FallbackCNN.
=============================================================

Usage
-----
    python training/train.py                        # train primary (GRAST)
    python training/train.py --model fallback       # train fallback CNN
    python training/train.py --horizon 1            # 1-step baseline
    python training/train.py --horizon 4            # progressive curriculum

Outputs
-------
    artifacts/primary_grast.pt       — primary model weights + config
    artifacts/fallback_cnn.pt        — fallback model weights + config
    artifacts/training_log.jsonl     — per-epoch metrics (append mode)

PRD references: §22 (methodology), §23 (loss), §24 (rollout), §24.1 (curriculum)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import DatasetReader, Transform, WindowIndex
from src.data.torch_dataset import ClimateWindowDataset
from src.features.geo import build_geo_features, N_GEO_CHANNELS
from src.models.grast import GRAST, FallbackCNN, C_IN
from src.rollout.engine import differentiable_rollout
from src.rollout.loss import MultiStepHuberLoss

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

LOG_PATH = ARTIFACTS / "training_log.jsonl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GRAST or FallbackCNN")
    p.add_argument("--model",    choices=["primary", "fallback"], default="primary")
    p.add_argument("--horizon",  type=int, default=4,
                   help="Autoregressive training horizon (1=one-step, 4/8/10=curriculum)")
    p.add_argument("--epochs",   type=int, default=40)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--workers",  type=int, default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    p.add_argument("--dropout",  type=float, default=0.05,
                   help="Dropout rate for primary model")
    p.add_argument("--base-ch",  type=int, default=32,
                   help="Base channel width (32=fast CPU ~350k params, 64=~1.4M params)")
    p.add_argument("--curriculum", action="store_true",
                   help="Run progressive curriculum: 1→2→4→horizon")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rmse_per_channel(pred_norm: list[torch.Tensor], tgt_norm: torch.Tensor) -> torch.Tensor:
    """
    Returns [horizon, 6] RMSE in normalized space.
    Lower = better.
    """
    horizon = min(len(pred_norm), tgt_norm.shape[1])
    result = []
    for k in range(horizon):
        diff = pred_norm[k] - tgt_norm[:, k]              # [B,6,H,W]
        rmse = diff.pow(2).mean(dim=(0, 2, 3)).sqrt()     # [6]
        result.append(rmse)
    return torch.stack(result, dim=0)                      # [horizon, 6]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    transform: Transform,
    geo_t: torch.Tensor,
    std_t: torch.Tensor,
    mean_t: torch.Tensor,
    device: torch.device,
    horizon: int,
    loss_fn: MultiStepHuberLoss,
) -> dict:
    model.eval()
    total_loss = 0.0
    rmse_accum = torch.zeros(horizon, 6, device=device)
    n_batches  = 0

    for x_norm, y_norm, doy_sin, doy_cos in loader:
        x_norm  = x_norm.to(device)      # [B,3,6,32,36]
        y_norm  = y_norm.to(device)      # [B,h,6,32,36]
        doy_sin = doy_sin.to(device)
        doy_cos = doy_cos.to(device)

        h = min(horizon, y_norm.shape[1])
        preds = differentiable_rollout(
            model, transform, geo_t, x_norm, doy_sin, doy_cos,
            steps=h, std_t=std_t, mean_t=mean_t,
        )

        loss = loss_fn(preds, y_norm[:, :h])
        total_loss += loss.item()
        rmse_accum[:h] += rmse_per_channel(preds, y_norm[:, :h])
        n_batches += 1

    if n_batches == 0:
        return {"val_loss": float("nan")}

    avg_loss = total_loss / n_batches
    avg_rmse = (rmse_accum / n_batches).cpu().numpy()      # [horizon, 6]

    metrics: dict = {"val_loss": avg_loss}
    channels = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
    for k in range(horizon):
        for ci, ch in enumerate(channels):
            metrics[f"rmse_d{k+1}_{ch}"] = float(avg_rmse[k, ci])
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_phase(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    transform: Transform,
    geo_t:    torch.Tensor,
    std_t:    torch.Tensor,
    mean_t:   torch.Tensor,
    device:   torch.device,
    n_epochs: int,
    horizon:  int,
    loss_fn:  MultiStepHuberLoss,
    phase_label: str,
) -> dict:
    """Train for n_epochs; return best val-loss metrics dict."""
    best_val_loss = math.inf
    best_state    = None
    best_metrics  = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for x_norm, y_norm, doy_sin, doy_cos in train_loader:
            x_norm  = x_norm.to(device)
            y_norm  = y_norm.to(device)
            doy_sin = doy_sin.to(device)
            doy_cos = doy_cos.to(device)

            h = min(horizon, y_norm.shape[1])

            optimizer.zero_grad()
            preds = differentiable_rollout(
                model, transform, geo_t, x_norm, doy_sin, doy_cos,
                steps=h, std_t=std_t, mean_t=mean_t,
            )
            loss = loss_fn(preds, y_norm[:, :h])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        n_train_batches = max(len(train_loader), 1)
        avg_train = train_loss / n_train_batches
        elapsed   = time.time() - t0

        # validation
        val_metrics = evaluate(
            model, val_loader, transform, geo_t,
            std_t, mean_t, device, horizon, loss_fn,
        )
        val_loss = val_metrics["val_loss"]

        log_entry = {
            "phase":      phase_label,
            "epoch":      epoch,
            "train_loss": avg_train,
            **val_metrics,
            "elapsed_s":  round(elapsed, 1),
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"[{phase_label}] epoch {epoch:03d}/{n_epochs:03d}  "
            f"train={avg_train:.4f}  val={val_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics  = dict(log_entry)

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # --- data ---
    print("Loading dataset …")
    reader    = DatasetReader()
    transform = Transform()

    std_t  = torch.from_numpy(transform.std).float().to(device)
    mean_t = torch.from_numpy(transform.mean).float().to(device)

    # geo features
    lsm, oro = reader.static[0], reader.static[1]
    geo_np = build_geo_features(lsm, oro, reader.lats, reader.lons)
    geo_t  = torch.from_numpy(geo_np).unsqueeze(0).float().to(device)   # [1,6,32,36]

    # splits
    train_idx = reader.split_indices("train")
    val_idx   = reader.split_indices("val")

    horizon = args.horizon
    train_win = WindowIndex(train_idx, n_inputs=3, max_horizon=horizon)
    val_win   = WindowIndex(val_idx,   n_inputs=3, max_horizon=horizon)

    train_ds = ClimateWindowDataset(reader, transform, train_win, horizon=horizon)
    val_ds   = ClimateWindowDataset(reader, transform, val_win,   horizon=horizon)

    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=False)

    # --- model ---
    if args.model == "primary":
        model = GRAST(c_in=C_IN, base_ch=args.base_ch, dropout=args.dropout).to(device)
        out_path = ARTIFACTS / "primary_grast.pt"
    else:
        model = FallbackCNN(c_in=C_IN, base_ch=args.base_ch).to(device)
        out_path = ARTIFACTS / "fallback_cnn.pt"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Parameters: {n_params:,}")

    # --- loss ---
    loss_fn = MultiStepHuberLoss(n_channels=6, horizon=horizon, beta=1.0).to(device)

    # --- optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # --- curriculum or single-phase training ---
    if args.curriculum and horizon > 1:
        # progressive: 1 → 2 → 4 → horizon
        phases = []
        h = 1
        while h < horizon:
            phases.append((h, max(args.epochs // 4, 5)))
            h = min(h * 2, horizon)
        phases.append((horizon, args.epochs - sum(e for _, e in phases)))
        phases = [(h, max(e, 5)) for h, e in phases]
    else:
        phases = [(horizon, args.epochs)]

    best_final = {}
    for phase_h, phase_epochs in phases:
        print(f"\n{'='*60}")
        print(f"Phase: horizon={phase_h}, epochs={phase_epochs}")
        print(f"{'='*60}")

        # rebuild loss for this horizon
        loss_fn = MultiStepHuberLoss(
            n_channels=6, horizon=phase_h, beta=1.0
        ).to(device)

        # rebuild windows if horizon changed
        if phase_h != horizon:
            pw_train = WindowIndex(train_idx, n_inputs=3, max_horizon=phase_h)
            pw_val   = WindowIndex(val_idx,   n_inputs=3, max_horizon=phase_h)
            pd_train = ClimateWindowDataset(reader, transform, pw_train, horizon=phase_h)
            pd_val   = ClimateWindowDataset(reader, transform, pw_val,   horizon=phase_h)
            pl_train = DataLoader(pd_train, batch_size=args.batch, shuffle=True,
                                  num_workers=args.workers)
            pl_val   = DataLoader(pd_val,   batch_size=args.batch, shuffle=False,
                                  num_workers=args.workers)
        else:
            pl_train, pl_val = train_loader, val_loader

        best_metrics = run_phase(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=pl_train,
            val_loader=pl_val,
            transform=transform,
            geo_t=geo_t,
            std_t=std_t,
            mean_t=mean_t,
            device=device,
            n_epochs=phase_epochs,
            horizon=phase_h,
            loss_fn=loss_fn,
            phase_label=f"h{phase_h}",
        )
        best_final = best_metrics

    # --- save artifact ---
    artifact = {
        "model_name":  f"{args.model}_grast" if args.model == "primary" else "fallback_cnn",
        "model_class": args.model,
        "c_in":        C_IN,
        "base_ch":     args.base_ch,
        "n_params":    n_params,
        "horizon":     horizon,
        "seed":        args.seed,
        "epochs":      args.epochs,
        "best_metrics": best_final,
        "state_dict":  model.state_dict(),
        "transform": {
            "mean": transform.mean.tolist(),
            "std":  transform.std.tolist(),
        },
        "geo_config": {
            "n_geo_channels": N_GEO_CHANNELS,
        },
    }

    torch.save(artifact, out_path)
    print(f"\nSaved: {out_path}")
    print(f"Best val_loss: {best_final.get('val_loss', 'N/A'):.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
