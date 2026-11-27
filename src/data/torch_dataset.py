"""
PyTorch Dataset wrapper for the WindowIndex.
Produces batches of (input_tensor, target_tensors) for training.
"""

from __future__ import annotations

from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import DatasetReader, Transform, WindowIndex, N_CHANNELS, GRID_H, GRID_W
from src.features.geo import build_geo_features, N_GEO_CHANNELS


class ClimateWindowDataset(Dataset):
    """
    Each item:
        x_norm    : [3, 6, 32, 36] normalized dynamic history
        y_norm    : [horizon, 6, 32, 36] normalized targets
        doy_sin   : float — sin of day-of-year for the last input day
        doy_cos   : float — cos of day-of-year for the last input day
    """

    def __init__(
        self,
        reader: DatasetReader,
        transform: Transform,
        window_index: WindowIndex,
        horizon: int = 10,
    ) -> None:
        self.reader = reader
        self.transform = transform
        self.window_index = window_index
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int):
        inp_days, tgt_days = self.window_index[idx]

        # --- dynamic history ---
        x_phys = self.reader.data[inp_days]            # [3, 6, 32, 36]
        x_norm = np.stack([self.transform.normalize(x_phys[i]) for i in range(3)], axis=0)

        # --- targets ---
        h = min(self.horizon, len(tgt_days))
        y_phys = self.reader.data[tgt_days[:h]]        # [h, 6, 32, 36]
        y_norm = np.stack([self.transform.normalize(y_phys[i]) for i in range(h)], axis=0)

        # --- seasonal phase for the last input day ---
        doy = self.reader.doy_sincos(inp_days[-1])     # [2]

        return (
            torch.from_numpy(x_norm).float(),          # [3, 6, 32, 36]
            torch.from_numpy(y_norm).float(),          # [h, 6, 32, 36]
            torch.tensor(doy[0], dtype=torch.float32),
            torch.tensor(doy[1], dtype=torch.float32),
        )
