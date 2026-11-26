"""Create a reproducible persistence/climatology report for one chronological partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_monsoon.baselines import CalendarDayClimatology
from deadlock_monsoon.contract import CHANNELS
from deadlock_monsoon.data import SplitSpec, chronological_masks
from deadlock_monsoon.dataset import EXPECTED_DATASET_SHA256, load_verified_dataset
from deadlock_monsoon.metrics import critical_success_index, latitude_weighted_rmse, spatial_variance_ratio


def evaluate(dataset_path: Path, partition: str) -> dict:
    dataset = load_verified_dataset(dataset_path)
    masks = chronological_masks(dataset.time, SplitSpec())
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test")
    climatology = CalendarDayClimatology.fit(dataset.data, dataset.time, masks["train"])
    leads = (1, 4, 8, 10)
    valid_last = np.flatnonzero(masks[partition])
    valid_last = valid_last[valid_last >= 2]
    valid_last = valid_last[valid_last + max(leads) < dataset.time.size]
    valid_last = valid_last[
        np.array([masks[partition][index + 1 : index + max(leads) + 1].all() for index in valid_last])
    ]
    report = {
        "schema_version": 1,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split": {
            "train_end": str(SplitSpec().train_end),
            "validation_end": str(SplitSpec().validation_end),
            "test_end": str(SplitSpec().test_end),
        },
        "partition": partition,
        "windows": int(valid_last.size),
        "leads": {},
    }
    for lead in leads:
        target_indices = valid_last + lead
        truth = dataset.data[target_indices]
        persistence = dataset.data[valid_last]
        clim = climatology.predict(dataset.time[target_indices])
        report["leads"][str(lead)] = {
            "persistence_rmse": dict(zip(CHANNELS, latitude_weighted_rmse(persistence, truth).tolist())),
            "climatology_rmse": dict(zip(CHANNELS, latitude_weighted_rmse(clim, truth).tolist())),
            "persistence_rain": critical_success_index(persistence[:, 1], truth[:, 1]),
            "climatology_rain": critical_success_index(clim[:, 1], truth[:, 1]),
            "persistence_variance_ratio": dict(
                zip(CHANNELS, spatial_variance_ratio(persistence, truth).tolist())
            ),
            "climatology_variance_ratio": dict(
                zip(CHANNELS, spatial_variance_ratio(clim, truth).tolist())
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.dataset, args.partition)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

