"""Bounded chronological GRAST candidate training and rollout-aware selection.

Outputs are local/ignored until the model passes the PRD release gates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_monsoon.data import ChannelTransform, SplitSpec, chronological_masks, window_indices
from deadlock_monsoon.dataset import EXPECTED_DATASET_SHA256, load_verified_dataset
from deadlock_monsoon.metrics import critical_success_index, latitude_weighted_rmse, spatial_variance_ratio
from deadlock_monsoon.model import GRAST, GRASTConfig, SemanticOutputMapper, parameter_count
from deadlock_monsoon.selection import is_better, score_checkpoint
from deadlock_monsoon.torch_rollout import autoregressive_rollout
from deadlock_monsoon.train_data import ForecastWindowDataset
from deadlock_monsoon.training import RainLossConfig, normalized_lead_weights, rollout_huber_loss

LEADS = (1, 4, 8, 10)


def choose_rows(rows: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count > len(rows):
        raise ValueError(f"requested {count} rows but only {len(rows)} are available")
    indices = np.random.default_rng(seed).choice(len(rows), size=count, replace=False)
    return rows[np.sort(indices)]


def evaluate(model, mapper, loader, mean, scale) -> tuple[dict, object]:
    predictions = []
    targets = []
    persistences = []
    occurrence_probabilities = []
    conditional_intensities = []
    has_occurrence_head = model.config.rain_mode == "occurrence_intensity"
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            result = autoregressive_rollout(
                model, mapper, batch["history"], batch["static"], batch["doy_sincos"],
                10, mean, scale, return_raw=has_occurrence_head,
            )
            if has_occurrence_head:
                output, raw = result
                occurrence_probabilities.append(torch.sigmoid(raw[:, :, 1]).numpy())
                conditional_intensities.append(torch.nn.functional.softplus(raw[:, :, 2]).numpy())
            else:
                output = result
            predictions.append(output.numpy())
            targets.append(batch["target"].numpy())
            persistences.append(batch["history"][:, -1].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    persistence = np.concatenate(persistences)
    occurrence_probability = (
        np.concatenate(occurrence_probabilities) if has_occurrence_head else None
    )
    conditional_intensity = (
        np.concatenate(conditional_intensities) if has_occurrence_head else None
    )
    finite = bool(np.isfinite(prediction).all())
    physical = bool(
        finite
        and np.min(prediction[:, :, 1]) >= 0
        and np.min(prediction[:, :, 0]) >= 233.15
        and np.max(prediction[:, :, 0]) <= 333.15
    )
    rmse = {}
    report = {"valid": physical, "leads": {}}
    for lead in LEADS:
        pred = prediction[:, lead - 1]
        truth = target[:, lead - 1]
        persist = persistence
        model_rmse = latitude_weighted_rmse(pred, truth)
        persistence_rmse = latitude_weighted_rmse(persist, truth)
        rmse[lead] = model_rmse
        report["leads"][str(lead)] = {
            "model_rmse": model_rmse.tolist(),
            "persistence_rmse": persistence_rmse.tolist(),
            "rain": critical_success_index(pred[:, 1], truth[:, 1]),
            "persistence_rain": critical_success_index(persist[:, 1], truth[:, 1]),
            "variance_ratio": spatial_variance_ratio(pred, truth).tolist(),
        }
        if has_occurrence_head:
            truth_event = truth[:, 1] > 10.0
            probability = occurrence_probability[:, lead - 1]
            intensity = conditional_intensity[:, lead - 1]
            report["leads"][str(lead)]["occurrence_head"] = {
                "mean_probability": float(probability.mean()),
                "target_event_rate": float(truth_event.mean()),
                "brier_score": float(np.mean((probability - truth_event) ** 2)),
                "mean_conditional_intensity_mm": float(intensity.mean()),
                "predicted_rain_event_rate": float((pred[:, 1] > 10.0).mean()),
            }
    day8_ratios = np.asarray(report["leads"]["8"]["variance_ratio"])
    passes_anti_blur = bool(np.isfinite(day8_ratios).all() and np.all(day8_ratios >= 0.40)
                            and np.all(day8_ratios <= 2.50))
    rain_scores = [report["leads"][str(lead)]["rain"]["csi"] for lead in (1, 4, 8)]
    passes_rain_activity = bool(all(np.isfinite(score) and score > 0 for score in rain_scores))
    false_alarm_pairs = [
        (
            report["leads"][str(lead)]["rain"]["false_alarm_ratio"],
            report["leads"][str(lead)]["persistence_rain"]["false_alarm_ratio"],
        )
        for lead in (1, 4, 8)
    ]
    passes_false_alarm = bool(all(
        np.isfinite(model_far)
        and np.isfinite(reference_far)
        and model_far <= min(0.75, reference_far + 0.10)
        for model_far, reference_far in false_alarm_pairs
    ))
    selection = score_checkpoint(
        rmse,
        scale.numpy(),
        np.asarray(report["leads"]["4"]["persistence_rmse"]),
        physical,
        passes_anti_blur=passes_anti_blur,
        passes_rain_activity=passes_rain_activity,
        passes_false_alarm=passes_false_alarm,
    )
    report["selection"] = asdict(selection)
    return report, selection


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    climate = load_verified_dataset(args.dataset)
    split = SplitSpec()
    masks = chronological_masks(climate.time, split)
    transform = ChannelTransform.fit(climate.data, masks["train"])
    train_rows = choose_rows(window_indices(masks["train"], 3, 10), args.train_windows, args.seed)
    validation_rows = choose_rows(
        window_indices(masks["validation"], 3, 10), args.validation_windows, args.seed + 1
    )
    train_data = ForecastWindowDataset(
        climate, train_rows, transform, horizon=10, geography_mode=args.geography_mode
    )
    validation_data = ForecastWindowDataset(
        climate, validation_rows, transform, horizon=10, geography_mode=args.geography_mode
    )
    train_rain_occurrence_prior = float((climate.data[masks["train"], 1] > 10.0).mean())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_data, batch_size=args.validation_batch_size, shuffle=False, num_workers=0
    )
    config = GRASTConfig(
        width=args.width,
        blocks=args.blocks,
        rain_mode=args.rain_mode,
        static_channels=train_data.static.shape[0],
        rain_occurrence_prior=train_rain_occurrence_prior,
    )
    model = GRAST(config)
    mapper = SemanticOutputMapper(torch.from_numpy(transform.scale), rain_mode=args.rain_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mean = torch.from_numpy(transform.mean)
    scale = torch.from_numpy(transform.scale)
    horizons = (2, 4, 8, 10)
    best_score = None
    best_epoch = None
    records = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists():
        args.checkpoint.unlink()  # prevent a rejected run from leaving a stale "best" artifact
    started = time.perf_counter()
    for epoch in range(args.epochs):
        horizon = horizons[min(epoch, len(horizons) - 1)]
        model.train()
        losses = []
        occurrence_losses = []
        tail_losses = []
        weights = normalized_lead_weights(horizon, args.lead_weights)
        rain_loss = RainLossConfig(
            occurrence_weight=args.rain_occurrence_weight,
            tail_weight=args.rain_tail_weight,
            positive_class_weight=args.rain_positive_class_weight,
        )
        for batch in train_loader:
            prediction, raw_outputs = autoregressive_rollout(
                model, mapper, batch["history"], batch["static"], batch["doy_sincos"],
                horizon, mean, scale, return_raw=True,
            )
            occurrence_logits = (
                raw_outputs[:, :, 1]
                if args.rain_mode == "occurrence_intensity"
                else None
            )
            breakdown = rollout_huber_loss(
                prediction,
                batch["target"][:, :horizon],
                scale,
                weights,
                rain=rain_loss,
                rain_occurrence_logits=occurrence_logits,
            )
            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(breakdown.total.detach()))
            occurrence_losses.append(float(breakdown.rain_occurrence.detach()))
            tail_losses.append(float(breakdown.rain_tail.detach()))
        validation, selection = evaluate(model, mapper, validation_loader, mean, scale)
        record = {
            "epoch": epoch + 1,
            "trained_horizon": horizon,
            "train_loss": float(np.mean(losses)),
            "rain_occurrence_loss": float(np.mean(occurrence_losses)),
            "rain_tail_loss": float(np.mean(tail_losses)),
            "validation": validation,
        }
        records.append(record)
        if is_better(selection, best_score):
            best_score = selection
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.checkpoint)
    return {
        "evidence_type": "chronological-candidate-prototype-not-release-candidate",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "seed": args.seed,
        "model_config": vars(config),
        "parameters": parameter_count(model),
        "train_windows": args.train_windows,
        "validation_windows": args.validation_windows,
        "lead_weights": args.lead_weights,
        "geography_mode": args.geography_mode,
        "rain_loss": {
            "occurrence_weight": args.rain_occurrence_weight,
            "tail_weight": args.rain_tail_weight,
            "positive_class_weight": args.rain_positive_class_weight,
            "threshold_mm": 10.0,
            "train_occurrence_prior": train_rain_occurrence_prior,
        },
        "best_epoch": best_epoch,
        "best_selection": None if best_score is None else asdict(best_score),
        "checkpoint": str(args.checkpoint),
        "epochs": records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/candidate.pt"))
    parser.add_argument("--train-windows", type=int, default=256)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--geography-mode", choices=("static", "metric"), default="static")
    parser.add_argument(
        "--rain-mode",
        choices=("softplus", "occurrence_intensity"),
        default="softplus",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lead-weights", choices=("uniform", "mild_increase"), default="uniform")
    parser.add_argument("--rain-occurrence-weight", type=float, default=0.0)
    parser.add_argument("--rain-tail-weight", type=float, default=0.0)
    parser.add_argument(
        "--rain-positive-class-weight",
        type=float,
        default=1.0,
        help="Fixed BCE positive weight established without validation/test data",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    safe_report = json_safe(report)
    args.report.write_text(
        json.dumps(safe_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "best_epoch": report["best_epoch"],
        "best_selection": safe_report["best_selection"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()

