"""Learned-only GRAST transition and physical semantic output mapping.

The transport gate is intentionally absent from this P0 model. It is a later,
time-boxed P1 ablation against this stable learned-only control.
"""

from __future__ import annotations

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
        self.config = config
        inputs = config.history * config.dynamic_channels + config.static_channels + 2
        self.encoder = nn.Conv2d(inputs, config.width, 3, padding=1, padding_mode="replicate")
        self.native = nn.Sequential(*[ResidualBlock(config.width) for _ in range(config.blocks)])
        self.coarse = nn.Sequential(ResidualBlock(config.width), ResidualBlock(config.width))
        self.fuse = nn.Conv2d(config.width * 2, config.width, 1)
        self.head = nn.Conv2d(config.width, config.dynamic_channels, 1)

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

    def __init__(self, tendency_scale: torch.Tensor):
        super().__init__()
        scale = torch.as_tensor(tendency_scale, dtype=torch.float32)
        if scale.shape != (6,) or torch.any(scale <= 0):
            raise ValueError("tendency_scale must contain six positive channel scales")
        self.register_buffer("tendency_scale", scale)

    def forward(self, raw: torch.Tensor, current_physical: torch.Tensor) -> torch.Tensor:
        if raw.shape != current_physical.shape or raw.shape[1:] != (6, 32, 36):
            raise ValueError("raw/current tensors must have shape [batch,6,32,36]")
        channels = []
        channels.append(233.15 + 100.0 * torch.sigmoid(raw[:, 0]))
        channels.append(F.softplus(raw[:, 1]))
        for channel in range(2, 6):
            channels.append(
                current_physical[:, channel]
                + raw[:, channel] * self.tendency_scale[channel]
            )
        return torch.stack(channels, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

