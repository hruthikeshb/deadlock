"""Deterministic real-data overfit prototype for the learned-only GRAST path.

This is a P0 engineering gate, not a competition checkpoint. It deliberately
trains a small model on a bounded subset and emits evidence about optimization,
validity, lead loss, and configuration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_monsoon.data import ChannelTransform, SplitSpec, chronological_masks, window_indices
from deadlock_monsoon.dataset import EXPECTED_DATASET_SHA256, load_verified_dataset
from deadlock_monsoon.model import GRAST, GRASTConfig, SemanticOutputMapper, parameter_count
from deadlock_monsoon.torch_rollout import autoregressive_rollout
from deadlock_monsoon.train_data import ForecastWindowDataset
from deadlock_monsoon.training import rollout_huber_loss


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    climate = load_verified_dataset(args.dataset)
    split = SplitSpec()
    masks = chronological_masks(climate.time, split)
    transform = ChannelTransform.fit(climate.data, masks["train"])
    rows = window_indices(masks["train"], history=3, horizon=args.horizon)
    if args.windows > rows.shape[0]:
        raise ValueError(f"requested {args.windows} windows but only {rows.shape[0]} exist")
    generator = np.random.default_rng(args.seed)
    selected = rows[generator.choice(rows.shape[0], size=args.windows, replace=False)]
    dataset = ForecastWindowDataset(climate, selected, transform, horizon=args.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    config = GRASTConfig(width=args.width, blocks=args.blocks)
    model = GRAST(config)
    mapper = SemanticOutputMapper(torch.from_numpy(transform.scale))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mapper.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    mean = torch.from_numpy(transform.mean)
    scale = torch.from_numpy(transform.scale)
    epoch_records = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        totals = []
        lead_totals = []
        channel_totals = []
        for batch in loader:
            prediction = autoregressive_rollout(
                model,
                mapper,
                batch["history"],
                batch["static"],
                batch["doy_sincos"],
                args.horizon,
                mean,
                scale,
            )
            breakdown = rollout_huber_loss(prediction, batch["target"], scale)
            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            totals.append(float(breakdown.total.detach()))
            lead_totals.append(breakdown.per_lead.detach().numpy())
            channel_totals.append(breakdown.per_channel.detach().numpy())
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(totals)),
                "per_lead": np.mean(lead_totals, axis=0).tolist(),
                "per_channel": np.mean(channel_totals, axis=0).tolist(),
            }
        )

    model.eval()
    with torch.inference_mode():
        sample = next(iter(DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)))
        output = autoregressive_rollout(
            model,
            mapper,
            sample["history"],
            sample["static"],
            sample["doy_sincos"],
            args.horizon,
            mean,
            scale,
        )
    validity = {
        "finite": bool(torch.isfinite(output).all()),
        "rain_min": float(output[:, :, 1].min()),
        "temperature_min": float(output[:, :, 0].min()),
        "temperature_max": float(output[:, :, 0].max()),
    }
    return {
        "evidence_type": "real-data-training-prototype-not-release-candidate",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split": {
            "train_end": str(split.train_end),
            "validation_end": str(split.validation_end),
            "test_end": str(split.test_end),
        },
        "seed": args.seed,
        "windows": args.windows,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "horizon": args.horizon,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "threads": args.threads,
        "model_config": vars(config),
        "parameters": parameter_count(model),
        "normalization_mean": transform.mean.tolist(),
        "normalization_scale": transform.scale.tolist(),
        "epochs_metrics": epoch_records,
        "loss_reduction_ratio": epoch_records[-1]["mean_loss"] / epoch_records[0]["mean_loss"],
        "validity": validity,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--horizon", type=int, choices=range(1, 11), default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "loss_reduction_ratio": report["loss_reduction_ratio"],
        "validity": report["validity"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()

