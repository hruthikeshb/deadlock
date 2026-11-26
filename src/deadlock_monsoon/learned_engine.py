"""Hash-verified CPU inference engine for a packaged GRAST bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_grast_bundle
from .torch_rollout import autoregressive_rollout
from .model import SemanticOutputMapper


class GRASTEngine:
    def __init__(self, bundle: str | Path, threads: int = 4):
        source = Path(bundle)
        self.model, self.manifest = load_grast_bundle(source, expected_role="primary")
        self.static = torch.from_numpy(np.load(source / "static.npy", allow_pickle=False))
        self.mean = torch.tensor(self.manifest.normalization_mean, dtype=torch.float32)
        self.scale = torch.tensor(self.manifest.normalization_scale, dtype=torch.float32)
        self.mapper = SemanticOutputMapper(
            self.scale,
            rain_mode=self.model.config.rain_mode,
        )
        torch.set_num_threads(threads)
        self.name = f"grast-primary-{self.manifest.code_commit[:7]}"

    def simulate(
        self,
        init_states: np.ndarray,
        doy_sincos: tuple[float, float],
        steps: int,
        perturbation: np.ndarray | None,
    ) -> np.ndarray:
        history = torch.from_numpy(np.asarray(init_states, dtype=np.float32)).unsqueeze(0)
        static = self.static.unsqueeze(0)
        seasonal = torch.tensor(doy_sincos, dtype=torch.float32).unsqueeze(0)
        shock = (
            None
            if perturbation is None
            else torch.from_numpy(np.asarray(perturbation, dtype=np.float32)).unsqueeze(0)
        )
        with torch.inference_mode():
            output = autoregressive_rollout(
                self.model,
                self.mapper,
                history,
                static,
                seasonal,
                steps,
                self.mean,
                self.scale,
                perturbation=shock,
            )
        return output.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
