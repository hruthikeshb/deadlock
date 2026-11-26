"""Export a retained raw checkpoint as a verified, self-describing bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_monsoon.checkpoint import export_grast_bundle, load_grast_bundle
from deadlock_monsoon.data import ChannelTransform, SplitSpec, chronological_masks
from deadlock_monsoon.dataset import load_verified_dataset
from deadlock_monsoon.geography import geography_features
from deadlock_monsoon.model import GRAST, GRASTConfig


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--role", choices=("primary", "fallback"), default="primary")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not report.get("best_selection", {}).get("release_eligible"):
        raise ValueError("refusing to export a report that did not pass every selection gate")
    climate = load_verified_dataset(args.dataset)
    split = SplitSpec()
    masks = chronological_masks(climate.time, split)
    transform = ChannelTransform.fit(climate.data, masks["train"])

    model = GRAST(GRASTConfig(**report["model_config"]))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    manifest = export_grast_bundle(
        args.output,
        model,
        role=args.role,
        normalization_mean=transform.mean,
        normalization_scale=transform.scale,
        split={
            "train_end": str(split.train_end),
            "validation_end": str(split.validation_end),
            "test_end": str(split.test_end),
        },
        training={
            "seed": report["seed"],
            "best_epoch": report["best_epoch"],
            "train_windows": report["train_windows"],
            "validation_windows": report["validation_windows"],
            "lead_weights": report["lead_weights"],
            "geography_mode": report["geography_mode"],
            "rain_loss": report["rain_loss"],
            "evidence_report": args.report.name,
        },
        code_commit=current_commit(),
        static_features=geography_features(
            climate.static,
            climate.lat,
            climate.lon,
            mode=report["geography_mode"],
        ),
    )
    restored, restored_manifest = load_grast_bundle(args.output, expected_role=args.role)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        if not torch.equal(expected, actual):
            raise RuntimeError("bundle round-trip changed model parameters")
    if restored_manifest != manifest:
        raise RuntimeError("bundle round-trip changed manifest metadata")
    if not np.isfinite(transform.mean).all() or not np.isfinite(transform.scale).all():
        raise RuntimeError("bundle normalization is nonfinite")
    print(json.dumps({
        "output": str(args.output),
        "role": manifest.role,
        "code_commit": manifest.code_commit,
        "weights_sha256": manifest.weights_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
