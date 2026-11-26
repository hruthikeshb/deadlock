"""Audit precipitation prevalence and persistence without validation leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_monsoon.data import SplitSpec, chronological_masks
from deadlock_monsoon.dataset import EXPECTED_DATASET_SHA256, load_verified_dataset


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize(rain: np.ndarray, land: np.ndarray) -> dict[str, object]:
    event = rain > 10.0
    previous = event[:-1]
    following = event[1:]
    return {
        "days": int(rain.shape[0]),
        "event_threshold_mm_day": 10.0,
        "event_rate_all": float(event.mean()),
        "event_rate_land": float(event[:, land].mean()),
        "event_rate_ocean": float(event[:, ~land].mean()),
        "zero_rate": float((rain <= 0.0).mean()),
        "mean_mm_day": float(rain.mean()),
        "quantiles_mm_day": {
            str(q): float(np.quantile(rain, q))
            for q in (0.5, 0.9, 0.95, 0.99, 0.999)
        },
        "event_given_event_previous_day": _safe_ratio(
            int(np.logical_and(following, previous).sum()), int(previous.sum())
        ),
        "event_given_dry_previous_day": _safe_ratio(
            int(np.logical_and(following, ~previous).sum()), int((~previous).sum())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    climate = load_verified_dataset(args.dataset)
    masks = chronological_masks(climate.time, SplitSpec())
    rain = climate.data[:, 1]
    land = climate.static[0] >= 0.5
    report = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "splits": {
            name: summarize(rain[mask], land) for name, mask in masks.items()
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
