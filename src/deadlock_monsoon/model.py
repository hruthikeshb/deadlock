"""Learned-only GRAST transition and physical semantic output mapping.

The transport gate is intentionally absent from this P0 model. It is a later,
time-boxed P1 ablation against this stable learned-only control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GRASTConfig:
    history: int = 3
    dynamic_channels: int = 6
    static_channels: int = 2
    width: int = 48
    blocks: int = 4
    rain_mode: str = "softplus"
    rain_occurrence_prior: float = 0.1
    # Absolute preserves schema-v2 bundles created before temperature-mode
    # metadata existed. New training explicitly requests bounded_residual.
    temperature_mode: str = "absolute"
    temperature_tendency_factor: float = 0.05


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate")
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(value)))
        return F.silu(value + self.norm2(self.conv2(hidden)))


class GRAST(nn.Module):
    """Compact nonperiodic, multiscale residual transition backbone."""

    def __init__(self, config: GRASTConfig = GRASTConfig()):
        super().__init__()
        if config.rain_mode not in {"softplus", "occurrence_intensity"}:
            raise ValueError("unsupported rain mode")
        if not 0.0 < config.rain_occurrence_prior < 1.0:
            raise ValueError("rain occurrence prior must be between zero and one")
        if config.temperature_mode not in {"absolute", "bounded_residual"}:
            raise ValueError("unsupported temperature mode")
        if config.temperature_tendency_factor <= 0:
            raise ValueError("temperature tendency factor must be positive")
        self.config = config
        inputs = config.history * config.dynamic_channels + config.static_channels + 2
        self.encoder = nn.Conv2d(inputs, config.width, 3, padding=1, padding_mode="replicate")
        self.native = nn.Sequential(*[ResidualBlock(config.width) for _ in range(config.blocks)])
        self.coarse = nn.Sequential(ResidualBlock(config.width), ResidualBlock(config.width))
        self.fuse = nn.Conv2d(config.width * 2, config.width, 1)
        output_channels = config.dynamic_channels + int(config.rain_mode == "occurrence_intensity")
        self.head = nn.Conv2d(config.width, output_channels, 1)
        if config.temperature_mode == "bounded_residual":
            with torch.no_grad():
                self.head.bias[0] = 0.0
        if config.rain_mode == "occurrence_intensity":
            prior_logit = math.log(config.rain_occurrence_prior / (1.0 - config.rain_occurrence_prior))
            with torch.no_grad():
                self.head.bias[1] = prior_logit

    def forward(
        self,
        history_normalized: torch.Tensor,
        static_features: torch.Tensor,
        doy_sincos: torch.Tensor,
    ) -> torch.Tensor:
        if history_normalized.ndim != 5 or history_normalized.shape[1:3] != (3, 6):
            raise ValueError("history must have shape [batch,3,6,lat,lon]")
        batch, _, _, height, width = history_normalized.shape
        if static_features.shape != (batch, self.config.static_channels, height, width):
            raise ValueError("static feature shape mismatch")
        if doy_sincos.shape != (batch, 2):
            raise ValueError("doy_sincos must have shape [batch,2]")
        dynamic = history_normalized.reshape(batch, -1, height, width)
        seasonal = doy_sincos[:, :, None, None].expand(-1, -1, height, width)
        encoded = F.silu(self.encoder(torch.cat((dynamic, static_features, seasonal), dim=1)))
        native = self.native(encoded)
        coarse = F.avg_pool2d(encoded, kernel_size=2)
        coarse = self.coarse(coarse)
        coarse = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
        return self.head(F.silu(self.fuse(torch.cat((native, coarse), dim=1))))


class SemanticOutputMapper(nn.Module):
    """Convert raw model outputs to physical next states without hard clipping."""

    def __init__(
        self,
        tendency_scale: torch.Tensor,
        rain_mode: str = "softplus",
        temperature_mode: str = "bounded_residual",
        temperature_tendency_factor: float = 0.05,
    ):
        super().__init__()
        scale = torch.as_tensor(tendency_scale, dtype=torch.float32)
        if scale.shape != (6,) or torch.any(scale <= 0):
            raise ValueError("tendency_scale must contain six positive channel scales")
        if rain_mode not in {"softplus", "occurrence_intensity"}:
            raise ValueError("unsupported rain mode")
        if temperature_mode not in {"absolute", "bounded_residual"}:
            raise ValueError("unsupported temperature mode")
        self.rain_mode = rain_mode
        self.temperature_mode = temperature_mode
        if temperature_tendency_factor <= 0:
            raise ValueError("temperature tendency factor must be positive")
        self.temperature_tendency_factor = float(temperature_tendency_factor)
        self.register_buffer("tendency_scale", scale)

    def forward(self, raw: torch.Tensor, current_physical: torch.Tensor) -> torch.Tensor:
        expected_raw_channels = 7 if self.rain_mode == "occurrence_intensity" else 6
        if raw.ndim != 4 or raw.shape[1:] != (expected_raw_channels, 32, 36):
            raise ValueError("raw tensor has the wrong semantic-head shape")
        if current_physical.ndim != 4 or current_physical.shape[1:] != (6, 32, 36):
            raise ValueError("current tensor must have shape [batch,6,32,36]")
        channels = []
        if self.temperature_mode == "absolute":
            temperature = 233.15 + 100.0 * torch.sigmoid(raw[:, 0])
        else:
            fraction = ((current_physical[:, 0] - 233.15) / 100.0).clamp(1e-6, 1 - 1e-6)
            current_logit = torch.logit(fraction)
            temperature = 233.15 + 100.0 * torch.sigmoid(
                current_logit + self.temperature_tendency_factor * raw[:, 0]
            )
        channels.append(temperature)
        if self.rain_mode == "softplus":
            channels.append(F.softplus(raw[:, 1]))
            offset = 0
        else:
            occurrence_probability = torch.sigmoid(raw[:, 1])
            intensity = F.softplus(raw[:, 2])
            channels.append(occurrence_probability * intensity)
            offset = 1
        for channel in range(2, 6):
            channels.append(
                current_physical[:, channel]
                + raw[:, channel + offset] * self.tendency_scale[channel]
            )
        return torch.stack(channels, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

