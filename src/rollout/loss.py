"""
Normalized Huber loss for multi-step autoregressive training (PRD §23).

L = Σ_k α_k · Σ_c w_c · mean_HW[ Huber((ŷ_{k,c} - y_{k,c}), β) ]

where both ŷ and y are in NORMALIZED space (so σ_c cancels out cleanly).
Weights w_c start equal; α_k starts uniform across the horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStepHuberLoss(nn.Module):
    """
    Parameters
    ----------
    n_channels : number of output channels (6).
    horizon    : maximum training horizon (number of steps).
    beta       : Huber threshold (transition from L1 to L2). Default 1.0.
    lead_weights : optional [horizon] tensor of α_k (defaults to uniform).
    chan_weights : optional [n_channels] tensor of w_c (defaults to equal).
    """

    def __init__(
        self,
        n_channels: int = 6,
        horizon: int = 10,
        beta: float = 1.0,
        lead_weights: torch.Tensor | None = None,
        chan_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.beta = beta

        # lead weights α_k — uniform by default
        if lead_weights is None:
            lead_weights = torch.ones(horizon) / horizon
        self.register_buffer("lead_weights", lead_weights / lead_weights.sum())

        # channel weights w_c — equal by default
        if chan_weights is None:
            chan_weights = torch.ones(n_channels) / n_channels
        self.register_buffer("chan_weights", chan_weights / chan_weights.sum())

    def forward(
        self,
        preds: list[torch.Tensor],   # list of [B, 6, H, W] normalized predictions
        targets: torch.Tensor,       # [B, horizon, 6, H, W] normalized targets
    ) -> torch.Tensor:
        """
        Returns scalar loss.
        preds[k] corresponds to targets[:, k, :, :, :].
        """
        total = torch.tensor(0.0, device=targets.device)
        horizon = min(len(preds), targets.shape[1])

        for k in range(horizon):
            pred_k = preds[k]                          # [B, 6, H, W]
            tgt_k  = targets[:, k]                    # [B, 6, H, W]

            # Huber loss per element
            err = F.huber_loss(pred_k, tgt_k, delta=self.beta, reduction="none")  # [B,6,H,W]

            # mean over batch, H, W → [6]
            err_per_ch = err.mean(dim=(0, 2, 3))                                   # [6]

            # weighted sum over channels
            step_loss = (self.chan_weights * err_per_ch).sum()

            total = total + self.lead_weights[k] * step_loss

        return total
