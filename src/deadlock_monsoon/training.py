"""Rollout-aware objectives and curriculum controls for P0 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    per_lead: torch.Tensor
    per_channel: torch.Tensor
    rain_occurrence: torch.Tensor
    rain_tail: torch.Tensor


@dataclass(frozen=True)
class RainLossConfig:
    threshold_mm: float = 10.0
    occurrence_weight: float = 0.0
    tail_weight: float = 0.0
    occurrence_temperature_mm: float = 2.0
    positive_class_weight: float = 1.0

    def validate(self) -> None:
        if self.threshold_mm <= 0 or self.occurrence_temperature_mm <= 0:
            raise ValueError("rain thresholds/temperature must be positive")
        if self.occurrence_weight < 0 or self.tail_weight < 0:
            raise ValueError("rain loss weights must be nonnegative")
        if self.positive_class_weight <= 0:
            raise ValueError("positive-class weight must be positive")


def normalized_lead_weights(horizon: int, mode: str = "uniform") -> torch.Tensor:
    if not 1 <= horizon <= 14:
        raise ValueError("horizon must be from 1 through 14")
    if mode == "uniform":
        weights = torch.ones(horizon)
    elif mode == "mild_increase":
        weights = torch.linspace(1.0, 1.5, horizon)
    else:
        raise ValueError("lead weighting must be 'uniform' or 'mild_increase'")
    return weights / weights.sum()


def rollout_huber_loss(
    prediction_physical: torch.Tensor,
    target_physical: torch.Tensor,
    channel_scale: torch.Tensor,
    lead_weights: torch.Tensor | None = None,
    beta: float = 1.0,
    rain: RainLossConfig | None = None,
    rain_occurrence_logits: torch.Tensor | None = None,
) -> LossBreakdown:
    """Channel-balanced physical error over every trained autoregressive lead."""
    if prediction_physical.shape != target_physical.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction_physical.ndim != 5 or prediction_physical.shape[2:] != (6, 32, 36):
        raise ValueError("rollouts must have shape [batch,lead,6,32,36]")
    scale = torch.as_tensor(
        channel_scale,
        dtype=prediction_physical.dtype,
        device=prediction_physical.device,
    )
    if scale.shape != (6,) or torch.any(scale <= 0) or not torch.isfinite(scale).all():
        raise ValueError("channel_scale must contain six positive finite values")
    normalized_error = (prediction_physical - target_physical) / scale.reshape(1, 1, 6, 1, 1)
    element = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        reduction="none",
        beta=beta,
    )
    per_lead_channel = element.mean(dim=(0, 3, 4))
    per_lead = per_lead_channel.mean(dim=1)
    per_channel = per_lead_channel.mean(dim=0)
    if lead_weights is None:
        weights = normalized_lead_weights(prediction_physical.shape[1]).to(
            device=prediction_physical.device,
            dtype=prediction_physical.dtype,
        )
    else:
        weights = torch.as_tensor(
            lead_weights,
            dtype=prediction_physical.dtype,
            device=prediction_physical.device,
        )
        if weights.shape != (prediction_physical.shape[1],) or torch.any(weights < 0):
            raise ValueError("lead_weights shape/value mismatch")
        if not torch.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("lead_weights must be finite with positive sum")
        weights = weights / weights.sum()
    base_total = (per_lead * weights).sum()
    rain_config = rain or RainLossConfig()
    rain_config.validate()
    predicted_rain = prediction_physical[:, :, 1]
    target_rain = target_physical[:, :, 1]
    event = (target_rain > rain_config.threshold_mm).to(prediction_physical.dtype)
    event_count = event.sum()
    # Keep this fixed across batches. A batch-derived inverse-frequency weight
    # changes the probability calibration as event prevalence fluctuates and
    # produced excess false alarms in long rollouts. The caller may supply a
    # value established from the training partition only; one is calibrated BCE.
    positive_weight = torch.as_tensor(
        rain_config.positive_class_weight,
        dtype=prediction_physical.dtype,
        device=prediction_physical.device,
    )
    if rain_occurrence_logits is None:
        occurrence_logits = (
            predicted_rain - rain_config.threshold_mm
        ) / rain_config.occurrence_temperature_mm
    else:
        occurrence_logits = torch.as_tensor(
            rain_occurrence_logits,
            dtype=prediction_physical.dtype,
            device=prediction_physical.device,
        )
        if occurrence_logits.shape != target_rain.shape:
            raise ValueError("rain occurrence logits must have shape [batch,lead,lat,lon]")
    occurrence_loss = F.binary_cross_entropy_with_logits(
        occurrence_logits,
        event,
        pos_weight=positive_weight,
    )
    tail_error = F.smooth_l1_loss(
        predicted_rain / scale[1],
        target_rain / scale[1],
        reduction="none",
        beta=beta,
    )
    tail_loss = (tail_error * event).sum() / torch.clamp(event_count, min=1.0)
    total = (
        base_total
        + rain_config.occurrence_weight * occurrence_loss
        + rain_config.tail_weight * tail_loss
    )
    return LossBreakdown(
        total=total,
        per_lead=per_lead,
        per_channel=per_channel,
        rain_occurrence=occurrence_loss,
        rain_tail=tail_loss,
    )


def curriculum_horizon(epoch: int, boundaries: tuple[int, int, int] = (3, 6, 10)) -> int:
    """Deterministic 2→4→8→10 curriculum; boundaries are logged configuration."""
    if epoch < 0 or len(boundaries) != 3 or tuple(sorted(boundaries)) != boundaries:
        raise ValueError("invalid epoch or curriculum boundaries")
    if epoch < boundaries[0]:
        return 2
    if epoch < boundaries[1]:
        return 4
    if epoch < boundaries[2]:
        return 8
    return 10

