"""Evaluate unfamiliar local shocks on validation-only atmospheric histories."""

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

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
LEADS = (1, 4, 8, 10)


def patch(center_y: int, center_x: int) -> np.ndarray:
    mask = np.zeros((32, 36), dtype=bool)
    y0, y1 = max(0, center_y - 1), min(32, center_y + 2)
    x0, x1 = max(0, center_x - 1), min(36, center_x + 2)
    mask[y0:y1, x0:x1] = True
    return mask


def location_masks(static: np.ndarray) -> dict[str, np.ndarray]:
    land_sea, orography = static
    coast_candidates = np.argwhere((land_sea > 0.1) & (land_sea < 0.9))
    coast_y, coast_x = coast_candidates[len(coast_candidates) // 2]
    mountain_y, mountain_x = np.unravel_index(np.argmax(orography), orography.shape)
    return {
        "interior": patch(16, 18),
        "north_boundary": patch(31, 18),
        "fractional_coast": patch(int(coast_y), int(coast_x)),
        "high_terrain": patch(int(mountain_y), int(mountain_x)),
    }


def make_shock(
    state: np.ndarray,
    channel: int,
    mask: np.ndarray,
    sign: int,
    magnitude_std: float,
    scale: np.ndarray,
) -> np.ndarray:
    shock = np.zeros((6, 32, 36), dtype=np.float32)
    amount = np.full(int(mask.sum()), magnitude_std * float(scale[channel]), dtype=np.float32)
    if channel == 1 and sign < 0:
        amount = np.minimum(amount, state[channel, mask] * 0.5)
    shock[channel, mask] = sign * amount
    return shock


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
    scale = engine.scale.numpy()
    places = location_masks(climate.static)
    scenarios = []
    skipped_insufficient_valid_amplitude = 0
    null_exact = True
    for anchor in anchors:
        history = climate.data[anchor - 2 : anchor + 1]
        seasonal = tuple(float(x) for x in doy_sincos(climate.time[anchor]))
        base = engine.simulate(history, seasonal, 10, None)
        zero = engine.simulate(history, seasonal, 10, np.zeros((6, 32, 36), dtype=np.float32))
        null_exact = null_exact and bool(np.array_equal(base, zero))
        for channel in range(6):
            for sign in (-1, 1):
                for magnitude in (0.25, 1.0):
                    for place, source in places.items():
                        shock = make_shock(history[-1], channel, source, sign, magnitude, scale)
                        input_amplitude_std = float(
                            np.max(np.abs(shock[channel]) / scale[channel])
                        )
                        # A negative-rain shock on a nearly dry patch cannot be
                        # made materially large without creating an invalid
                        # atmospheric input. Record and skip that inapplicable case.
                        if input_amplitude_std < 1e-2:
                            skipped_insufficient_valid_amplitude += 1
                            continue
                        active = np.any(shock != 0, axis=0)
                        if not active.any():
                            continue
                        perturbed = engine.simulate(history, seasonal, 10, shock)
                        diagnostic = perturbation_report(base, perturbed, scale, active)
                        physical = bool(
                            np.isfinite(perturbed).all()
                            and np.min(perturbed[:, 1]) >= 0
                            and np.min(perturbed[:, 0]) >= 233.15
                            and np.max(perturbed[:, 0]) <= 333.15
                        )
                        scenarios.append({
                            "date": str(climate.time[anchor]),
                            "channel": CHANNELS[channel],
                            "sign": sign,
                            "magnitude_std": magnitude,
                            "location": place,
                            "input_amplitude_std": input_amplitude_std,
                            "physical": physical,
                            "diagnostic": diagnostic,
                            "lead_summary": {
                                str(lead): {
                                    "near_fraction": diagnostic["near_fraction"][lead - 1],
                                    "radius_cells": diagnostic["radius_cells"][lead - 1],
                                }
                                for lead in LEADS
                            },
                        })
    summary = {
        "scenario_count": len(scenarios),
        "skipped_insufficient_valid_amplitude": skipped_insufficient_valid_amplitude,
        "null_exact": null_exact,
        "all_physical": all(item["physical"] for item in scenarios),
        "all_finite": all(item["diagnostic"]["finite"] for item in scenarios),
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
