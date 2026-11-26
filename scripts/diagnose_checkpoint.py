"""Validation-only drift diagnostics on a fixed, seed-independent window set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_monsoon.baselines import CalendarDayClimatology
from deadlock_monsoon.data import SplitSpec, chronological_masks, window_indices
from deadlock_monsoon.dataset import load_verified_dataset
from deadlock_monsoon.learned_engine import GRASTEngine
from deadlock_monsoon.metrics import spatial_variance_ratio
from deadlock_monsoon.train_data import doy_sincos
from train_candidate import choose_rows

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
LEADS = (1, 4, 8, 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-windows", type=int, default=256)
    parser.add_argument("--validation-seed", type=int, default=20261126)
    args = parser.parse_args()

    climate = load_verified_dataset(args.dataset)
    masks = chronological_masks(climate.time, SplitSpec())
    rows = choose_rows(
        window_indices(masks["validation"], 3, 10),
        args.validation_windows,
        args.validation_seed,
    )
    climatology = CalendarDayClimatology.fit(climate.data, climate.time, masks["train"])
    engine = GRASTEngine(args.bundle)
    predictions = []
    for row in rows:
        predictions.append(engine.simulate(
            climate.data[row[:3]],
            tuple(float(x) for x in doy_sincos(climate.time[row[2]])),
            10,
            None,
        ))
    prediction = np.stack(predictions)
    target = climate.data[rows[:, 3:]]
    target_times = climate.time[rows[:, 3:]]
    climatology_prediction = climatology.predict(target_times.reshape(-1)).reshape(target.shape)

    diagnostics = {}
    for lead in LEADS:
        index = lead - 1
        pred_error = prediction[:, index] - target[:, index]
        clim_error = climatology_prediction[:, index] - target[:, index]
        months = np.array([
            int(np.datetime_as_string(date, unit="D")[5:7]) for date in target_times[:, index]
        ])
        diagnostics[str(lead)] = {
            "model_bias": dict(zip(CHANNELS, pred_error.mean(axis=(0, 2, 3)).tolist())),
            "climatology_bias": dict(zip(CHANNELS, clim_error.mean(axis=(0, 2, 3)).tolist())),
            "model_spatial_variance_ratio": dict(zip(
                CHANNELS, spatial_variance_ratio(prediction[:, index], target[:, index]).tolist()
            )),
            "climatology_spatial_variance_ratio": dict(zip(
                CHANNELS,
                spatial_variance_ratio(climatology_prediction[:, index], target[:, index]).tolist(),
            )),
            "seasonal_model_bias": {
                season: dict(zip(CHANNELS, pred_error[np.isin(months, season_months)].mean(
                    axis=(0, 2, 3)
                ).tolist()))
                for season, season_months in {
                    "DJF": (12, 1, 2), "MAM": (3, 4, 5),
                    "JJAS": (6, 7, 8, 9), "ON": (10, 11),
                }.items()
            },
        }
    report = {
        "partition": "validation",
        "locked_test_opened": False,
        "validation_seed": args.validation_seed,
        "validation_windows": args.validation_windows,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
