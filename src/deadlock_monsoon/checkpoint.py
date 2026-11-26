"""Hash-verified, state-dict-only model artifact bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import CHANNELS
from .dataset import EXPECTED_DATASET_SHA256
from .model import GRAST, GRASTConfig

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    role: str
    model_class: str
    model_config: dict[str, Any]
    channel_order: list[str]
    dataset_sha256: str
    normalization_mean: list[float]
    normalization_scale: list[float]
    split: dict[str, str]
    training: dict[str, Any]
    code_commit: str
    weights_sha256: str
    static_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_grast_bundle(
    directory: str | Path,
    model: GRAST,
    role: str,
    normalization_mean: np.ndarray,
    normalization_scale: np.ndarray,
    split: dict[str, str],
    training: dict[str, Any],
    code_commit: str,
    static_features: np.ndarray,
) -> ArtifactManifest:
    if role not in {"primary", "fallback"}:
        raise ValueError("role must be primary or fallback")
    mean = np.asarray(normalization_mean, dtype=np.float32)
    scale = np.asarray(normalization_scale, dtype=np.float32)
    if mean.shape != (6,) or scale.shape != (6,) or np.any(scale <= 0):
        raise ValueError("normalization arrays must contain six valid channels")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    weights_path = target / "weights.pt"
    static_path = target / "static.npy"
    static = np.asarray(static_features, dtype=np.float32)
    if static.shape != (model.config.static_channels, 32, 36) or not np.isfinite(static).all():
        raise ValueError("static features do not match model configuration")
    torch.save(model.state_dict(), weights_path)
    np.save(static_path, static, allow_pickle=False)
    manifest = ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        role=role,
        model_class="GRAST",
        model_config=asdict(model.config),
        channel_order=list(CHANNELS),
        dataset_sha256=EXPECTED_DATASET_SHA256,
        normalization_mean=mean.tolist(),
        normalization_scale=scale.tolist(),
        split=dict(split),
        training=dict(training),
        code_commit=code_commit,
        weights_sha256=_sha256(weights_path),
        static_sha256=_sha256(static_path),
    )
    (target / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_grast_bundle(directory: str | Path, expected_role: str | None = None):
    source = Path(directory)
    manifest_data = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest = ArtifactManifest(**manifest_data)
    if manifest.schema_version != SCHEMA_VERSION or manifest.model_class != "GRAST":
        raise ValueError("unsupported artifact schema/model")
    if expected_role is not None and manifest.role != expected_role:
        raise ValueError("artifact role mismatch")
    if tuple(manifest.channel_order) != CHANNELS:
        raise ValueError("artifact channel order mismatch")
    if manifest.dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError("artifact dataset provenance mismatch")
    weights_path = source / "weights.pt"
    static_path = source / "static.npy"
    if _sha256(weights_path) != manifest.weights_sha256:
        raise ValueError("artifact weights hash mismatch")
    if _sha256(static_path) != manifest.static_sha256:
        raise ValueError("artifact static-feature hash mismatch")
    model = GRAST(GRASTConfig(**manifest.model_config))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    static = np.load(static_path, allow_pickle=False)
    if static.shape != (model.config.static_channels, 32, 36) or not np.isfinite(static).all():
        raise ValueError("artifact static features are invalid")
    return model, manifest

