"""Evaluate the exact warm-ocean, monsoon-stall, and null contract shocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_monsoon.data import SplitSpec, chronological_masks
from deadlock_monsoon.dataset import load_verified_dataset
from deadlock_monsoon.diagnostics import perturbation_report
from deadlock_monsoon.learned_engine import GRASTEngine
from deadlock_monsoon.train_data import doy_sincos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=4)
    args = parser.parse_args()

    climate = load_verified_dataset(args.dataset)
    masks = chronological_masks(climate.time, SplitSpec())
    validation = np.flatnonzero(masks["validation"])
    anchors = validation[np.linspace(2, len(validation) - 11, args.histories, dtype=int)]
    engine = GRASTEngine(args.bundle)
    lat, lon = np.meshgrid(climate.lat, climate.lon, indexing="ij")
    bay = (lat >= 8) & (lat <= 20) & (lon >= 82) & (lon <= 94)
    arabian = (lat >= 8) & (lat <= 20) & (lon >= 65) & (lon <= 75)
    scenarios = []
    null_exact = True
    for anchor in anchors:
        history = climate.data[anchor - 2:anchor + 1]
        seasonal = tuple(float(x) for x in doy_sincos(climate.time[anchor]))
        base = engine.simulate(history, seasonal, 10, None)
        zero = engine.simulate(history, seasonal, 10, np.zeros((6, 32, 36), dtype=np.float32))
        null_exact = null_exact and bool(np.array_equal(base, zero))
        shocks = {}
        warm = np.zeros((6, 32, 36), dtype=np.float32)
        warm[0, bay] = 3.0
        warm[5, bay] = 0.15 * history[-1, 5, bay]
        shocks["warm_ocean"] = (warm, bay)
        stall = np.zeros((6, 32, 36), dtype=np.float32)
        stall[2, arabian] = -0.40 * history[-1, 2, arabian]
        stall[3, arabian] = -0.40 * history[-1, 3, arabian]
        shocks["monsoon_stall"] = (stall, arabian)
        for name, (shock, source) in shocks.items():
            perturbed = engine.simulate(history, seasonal, 10, shock)
            diagnostic = perturbation_report(base, perturbed, engine.scale.numpy(), source)
            scenarios.append({
                "date": str(climate.time[anchor]),
                "shock": name,
                "physical": bool(
                    np.isfinite(perturbed).all()
                    and np.min(perturbed[:, 1]) >= 0
                    and np.min(perturbed[:, 0]) >= 233.15
                    and np.max(perturbed[:, 0]) <= 333.15
                ),
                "diagnostic": diagnostic,
            })
    summary = {
        "scenario_count": len(scenarios),
        "null_exact": null_exact,
        "all_physical": all(item["physical"] for item in scenarios),
        "all_nonzero": all(item["diagnostic"]["nonzero"] for item in scenarios),
        "all_stable": all(item["diagnostic"]["stable"] for item in scenarios),
        "minimum_day1_near_fraction": min(
            item["diagnostic"]["near_fraction"][0] for item in scenarios
        ),
        "maximum_response_std": max(
            item["diagnostic"]["max_response_std"] for item in scenarios
        ),
    }
    report = {"summary": summary, "scenarios": scenarios}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
