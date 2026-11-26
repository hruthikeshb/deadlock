"""Differentiable shared-weight autoregressive rollout for training and export."""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import SemanticOutputMapper


def advance_doy(pair: torch.Tensor) -> torch.Tensor:
    angle = 2.0 * math.pi / 365.0
    sine, cosine = pair[:, 0], pair[:, 1]
    return torch.stack(
        (sine * math.cos(angle) + cosine * math.sin(angle),
         cosine * math.cos(angle) - sine * math.sin(angle)),
        dim=1,
    )


def autoregressive_rollout(
    model: nn.Module,
    mapper: SemanticOutputMapper,
    init_states: torch.Tensor,
    static_features: torch.Tensor,
    doy_sincos: torch.Tensor,
    steps: int,
    normalization_mean: torch.Tensor,
    normalization_scale: torch.Tensor,
    perturbation: torch.Tensor | None = None,
) -> torch.Tensor:
    if init_states.ndim != 5 or init_states.shape[1:] != (3, 6, 32, 36):
        raise ValueError("init_states must have shape [batch,3,6,32,36]")
    if not 1 <= steps <= 14:
        raise ValueError("steps must be from 1 through 14")
    history = init_states.clone()
    if perturbation is not None:
        if perturbation.shape != history[:, -1].shape:
            raise ValueError("perturbation shape mismatch")
        history[:, -1] = history[:, -1] + perturbation
    mean = normalization_mean.reshape(1, 1, 6, 1, 1)
    scale = normalization_scale.reshape(1, 1, 6, 1, 1)
    if torch.any(scale <= 0):
        raise ValueError("normalization scales must be positive")
    pair = doy_sincos
    outputs = []
    for _ in range(steps):
        history_normalized = (history - mean) / scale
        raw = model(history_normalized, static_features, pair)
        next_state = mapper(raw, history[:, -1])
        if not torch.isfinite(next_state).all():
            raise RuntimeError("model produced nonfinite next state")
        outputs.append(next_state)
        history = torch.cat((history[:, 1:], next_state[:, None]), dim=1)
        pair = advance_doy(pair)
    return torch.stack(outputs, dim=1)

