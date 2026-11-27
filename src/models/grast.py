"""
GRAST — Geography-aware Residual Atmospheric State Transition
=============================================================
Primary candidate model (PRD §17).

Input tensor shape:  [B, C_in, 32, 36]
  where C_in = 3*6 (dynamic history) + N_GEO_CHANNELS (static/geo) + 2 (doy_sincos)
             = 18 + 6 + 2 = 26

Output tensor shape: [B, 6, 32, 36]  — normalized per-channel tendencies Δz

The model predicts residuals (tendencies), NOT full next states.
The rollout engine applies:  z_{t+1} = z_t + r ⊙ Δz_t

Fallback model (FallbackCNN) is a simpler single-scale variant with the same
input/output contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 6
N_HISTORY  = 3
N_GEO      = 6
N_DOY      = 2
C_IN       = N_HISTORY * N_CHANNELS + N_GEO + N_DOY   # 26


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class NonPeriodicPad(nn.Module):
    """Replicate-pad on all 4 spatial sides (non-periodic / open boundaries)."""

    def __init__(self, pad: int = 1) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.pad,) * 4, mode="replicate")


class ResBlock(nn.Module):
    """
    Residual block:  Conv → BN → ReLU → Conv → BN → (+skip)
    Uses NonPeriodicPad to avoid artificial periodic boundary.
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pad = NonPeriodicPad(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=0, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.pad(x))
        h = F.relu(self.bn1(h), inplace=True)
        h = self.drop(h)
        h = self.conv2(self.pad(h))
        h = self.bn2(h)
        return F.relu(h + x, inplace=True)


class DownBlock(nn.Module):
    """Strided conv to coarser scale (factor 2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pad  = NonPeriodicPad(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(self.pad(x))), inplace=True)


class UpBlock(nn.Module):
    """Bilinear upsample + conv to merge coarse features back to fine scale."""

    def __init__(self, coarse_ch: int, fine_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pad  = NonPeriodicPad(1)
        self.conv = nn.Conv2d(coarse_ch + fine_ch, out_ch, 3, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(
        self,
        coarse: torch.Tensor,
        fine: torch.Tensor,
    ) -> torch.Tensor:
        up = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        h  = torch.cat([up, fine], dim=1)
        return F.relu(self.bn(self.conv(self.pad(h))), inplace=True)


# ---------------------------------------------------------------------------
# GRAST  (primary model, ~1.5–3M parameters)
# ---------------------------------------------------------------------------

class GRAST(nn.Module):
    """
    Geography-aware Residual Atmospheric State Transition.

    Architecture:
        Encoder (shallow conv)  →  Fine-scale residual blocks
                                →  DownBlock  →  Coarse residual blocks
                                →  UpBlock (skip fusion)
                                →  Fine residual blocks
                                →  Output head (tendency + precipitation clamp)
    """

    def __init__(
        self,
        c_in:         int   = C_IN,          # 26
        base_ch:      int   = 64,
        n_fine_pre:   int   = 2,
        n_coarse:     int   = 3,
        n_fine_post:  int   = 2,
        dropout:      float = 0.05,
    ) -> None:
        super().__init__()
        coarse_ch = base_ch * 2

        # --- encoder ---
        self.encoder = nn.Sequential(
            NonPeriodicPad(1),
            nn.Conv2d(c_in, base_ch, 3, padding=0, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # --- fine-scale blocks ---
        self.fine_pre  = nn.Sequential(*[ResBlock(base_ch, dropout) for _ in range(n_fine_pre)])

        # --- coarse branch ---
        self.down      = DownBlock(base_ch, coarse_ch)
        self.coarse    = nn.Sequential(*[ResBlock(coarse_ch, dropout) for _ in range(n_coarse)])
        self.up        = UpBlock(coarse_ch, base_ch, base_ch)

        # --- fine post-merge blocks ---
        self.fine_post = nn.Sequential(*[ResBlock(base_ch, dropout) for _ in range(n_fine_post)])

        # --- output head: predict 6-channel tendency ---
        self.head = nn.Sequential(
            NonPeriodicPad(1),
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            NonPeriodicPad(1),
            nn.Conv2d(base_ch // 2, N_CHANNELS, 3, padding=0, bias=True),
        )

        # --- learnable per-channel residual scale (initialized to small value) ---
        self.residual_scale = nn.Parameter(torch.ones(N_CHANNELS) * 0.1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C_in, 32, 36]
        Returns tendency Δz : [B, 6, 32, 36]
        """
        enc   = self.encoder(x)                 # [B, base_ch, 32, 36]
        fine0 = self.fine_pre(enc)               # [B, base_ch, 32, 36]
        coarse = self.down(fine0)               # [B, coarse_ch, 16, 18]
        coarse = self.coarse(coarse)             # [B, coarse_ch, 16, 18]
        merged = self.up(coarse, fine0)          # [B, base_ch, 32, 36]
        feat   = self.fine_post(merged)          # [B, base_ch, 32, 36]
        dz     = self.head(feat)                 # [B, 6, 32, 36]

        # apply per-channel learnable scale
        r = self.residual_scale.view(1, N_CHANNELS, 1, 1)
        return dz * r


# ---------------------------------------------------------------------------
# FallbackCNN  (simpler single-scale, ~0.5M params)
# ---------------------------------------------------------------------------

class FallbackCNN(nn.Module):
    """
    Smaller single-scale residual CNN.  Same input/output contract as GRAST.
    Trained independently and validated separately (PRD §17 fallback).
    """

    def __init__(
        self,
        c_in:    int   = C_IN,
        base_ch: int   = 32,
        n_res:   int   = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            NonPeriodicPad(1),
            nn.Conv2d(c_in, base_ch, 3, padding=0, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(base_ch, dropout) for _ in range(n_res)])
        self.head = nn.Sequential(
            NonPeriodicPad(1),
            nn.Conv2d(base_ch, N_CHANNELS, 3, padding=0, bias=True),
        )
        self.residual_scale = nn.Parameter(torch.ones(N_CHANNELS) * 0.1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc  = self.encoder(x)
        feat = self.res_blocks(enc)
        dz   = self.head(feat)
        r    = self.residual_scale.view(1, N_CHANNELS, 1, 1)
        return dz * r
