"""
RolloutEngine
=============
Shared autoregressive rollout used at both training and inference time (PRD §24).

Key contracts:
  - Perturbation δ is applied exactly once to the last input state.
  - The same model weights are used for every step.
  - Seasonal phase is advanced analytically from the supplied doy_sincos.
  - Precipitation is clamped non-negative in physical space at every step.
  - Returns denormalized physical states [steps, 6, 32, 36].
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import Transform, N_CHANNELS, GRID_H, GRID_W
from src.features.geo import N_GEO_CHANNELS

RAIN_CH = 1   # channel index for precipitation (tp)


def _advance_doy_sincos(sin_val: float, cos_val: float) -> tuple[float, float]:
    """
    Advance seasonal phase by exactly one day.

    Uses the angle-addition formula:
        angle_{t+1} = angle_t + 2π/365
    so  sin_{t+1} = sin_t·cos(Δ) + cos_t·sin(Δ)
        cos_{t+1} = cos_t·cos(Δ) - sin_t·sin(Δ)
    """
    delta = 2.0 * np.pi / 365.0
    cd, sd = np.cos(delta), np.sin(delta)
    new_sin = sin_val * cd + cos_val * sd
    new_cos = cos_val * cd - sin_val * sd
    return float(new_sin), float(new_cos)


class RolloutEngine:
    """
    Wraps a trained model for autoregressive rollout.

    Parameters
    ----------
    model      : GRAST or FallbackCNN (any nn.Module with same forward signature).
    transform  : Transform object (normalize / denormalize).
    geo_tensor : np.ndarray [N_GEO_CHANNELS, 32, 36] — static geo features.
    device     : torch device for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: Transform,
        geo_tensor: np.ndarray,
        device: torch.device,
    ) -> None:
        self.model = model
        self.transform = transform
        self.device = device

        # Pre-convert static geo to a device tensor [1, N_GEO, H, W]
        self._geo_t = torch.from_numpy(geo_tensor).unsqueeze(0).to(device)

    @torch.no_grad()
    def rollout(
        self,
        init_states: np.ndarray,          # [3, 6, 32, 36] physical units, oldest first
        doy_sincos: np.ndarray,            # [2] seasonal phase for init_states[-1]
        steps: int,
        perturbation: Optional[np.ndarray] = None,  # [6, 32, 36] or None
    ) -> np.ndarray:
        """
        Return physical rollout [steps, 6, 32, 36].
        """
        assert 1 <= steps <= 14, f"steps must be 1–14, got {steps}"
        assert init_states.shape == (3, N_CHANNELS, GRID_H, GRID_W)

        self.model.eval()
        tf = self.transform

        # --- copy inputs; apply perturbation once to last state ---
        states = init_states.copy()
        if perturbation is not None:
            states[-1] = states[-1] + perturbation  # exactly once

        # normalize history → [3, 6, 32, 36]
        norm_history = np.stack([tf.normalize(s) for s in states], axis=0)

        sin_val, cos_val = float(doy_sincos[0]), float(doy_sincos[1])
        outputs_phys = []

        for _ in range(steps):
            # build input tensor [1, 26, 32, 36]
            doy_map = np.array([[sin_val], [cos_val]], dtype="float32")   # [2,1]
            doy_map = np.broadcast_to(doy_map[:, :, None],
                                      (2, GRID_H, GRID_W))                # [2,32,36]

            inp = np.concatenate([
                norm_history.reshape(3 * N_CHANNELS, GRID_H, GRID_W),     # [18,32,36]
                self._geo_t.squeeze(0).cpu().numpy(),                      # [6,32,36]
                doy_map,                                                   # [2,32,36]
            ], axis=0)                                                     # [26,32,36]

            x = torch.from_numpy(inp[None]).to(self.device)                # [1,26,32,36]

            # predict tendency
            dz = self.model(x)                                             # [1,6,32,36]
            dz_np = dz.squeeze(0).cpu().numpy()                            # [6,32,36]

            # residual update on the last normalized state
            z_next = norm_history[-1] + dz_np                              # [6,32,36]

            # denormalize
            x_next = tf.denormalize(z_next)                                # [6,32,36]

            # enforce physics constraints
            x_next[RAIN_CH] = np.maximum(x_next[RAIN_CH], 0.0)            # non-neg rain
            assert np.isfinite(x_next).all(), "non-finite output at rollout step"

            outputs_phys.append(x_next)

            # --- advance history ---
            norm_history = np.concatenate([
                norm_history[1:],                                           # drop oldest
                tf.normalize(x_next)[None],                                # append new
            ], axis=0)

            # --- advance seasonal phase ---
            sin_val, cos_val = _advance_doy_sincos(sin_val, cos_val)

        return np.stack(outputs_phys, axis=0)                              # [steps,6,32,36]


# ---------------------------------------------------------------------------
# Training-time differentiable rollout (returns tensors, supports grad)
# ---------------------------------------------------------------------------

def differentiable_rollout(
    model: nn.Module,
    transform: Transform,
    geo_t: torch.Tensor,          # [1, N_GEO, H, W] on device
    norm_history_t: torch.Tensor, # [B, 3, 6, H, W] normalized, on device
    doy_sin: torch.Tensor,        # [B]
    doy_cos: torch.Tensor,        # [B]
    steps: int,
    std_t: torch.Tensor,          # [6] channel stds on device (for denorm rain clamp)
    mean_t: torch.Tensor,         # [6] channel means on device
    rain_clamp: bool = True,
) -> list[torch.Tensor]:
    """
    Differentiable autoregressive rollout for training.

    Returns list of `steps` tensors, each [B, 6, H, W] in NORMALIZED space.
    """
    B, _, _, H, W = norm_history_t.shape
    device = norm_history_t.device
    delta = 2.0 * np.pi / 365.0
    cd = torch.cos(torch.tensor(delta, device=device))
    sd = torch.sin(torch.tensor(delta, device=device))

    history = norm_history_t.clone()   # [B, 3, 6, H, W]
    sin_v = doy_sin.clone()            # [B]
    cos_v = doy_cos.clone()            # [B]

    preds_norm = []

    for _ in range(steps):
        # doy map broadcast to [B, 2, H, W]
        doy_map = torch.stack([sin_v, cos_v], dim=1)       # [B, 2]
        doy_map = doy_map[:, :, None, None].expand(B, 2, H, W)

        # geo broadcast to [B, N_GEO, H, W]
        geo_b = geo_t.expand(B, -1, -1, -1)

        # flatten history: [B, 3*6, H, W]
        hist_flat = history.reshape(B, 3 * N_CHANNELS, H, W)

        inp = torch.cat([hist_flat, geo_b, doy_map], dim=1)    # [B, 26, H, W]

        dz = model(inp)                                         # [B, 6, H, W]

        z_next = history[:, -1] + dz                           # [B, 6, H, W]

        if rain_clamp:
            # clamp in normalized space: rain physical ≥ 0  → z_rain ≥ -mean[1]/std[1]
            rain_norm_min = -mean_t[1] / std_t[1]
            z_next_clamped = z_next.clone()
            z_next_clamped[:, RAIN_CH] = z_next[:, RAIN_CH].clamp(min=rain_norm_min.item())
            z_next = z_next_clamped

        preds_norm.append(z_next)

        # advance history
        history = torch.cat([history[:, 1:], z_next.unsqueeze(1)], dim=1)

        # advance doy analytically
        new_sin = sin_v * cd + cos_v * sd
        new_cos = cos_v * cd - sin_v * sd
        sin_v, cos_v = new_sin, new_cos

    return preds_norm    # list of [B, 6, H, W] in normalized space
