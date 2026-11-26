"""Torch datasets built from leakage-free chronological NetCDF windows."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import ChannelTransform
from .dataset import ClimateDataset


def doy_sincos(date: np.datetime64) -> np.ndarray:
    day = np.datetime64(date, "D")
    year = int(np.datetime_as_string(day, unit="Y"))
    day_of_year = int((day - np.datetime64(f"{year:04d}-01-01", "D")).astype(int)) + 1
    angle = 2.0 * math.pi * day_of_year / 365.0
    return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)


def normalized_static(static: np.ndarray) -> np.ndarray:
    value = np.asarray(static, dtype=np.float32).copy()
    if value.shape != (2, 32, 36):
        raise ValueError("static fields must have shape [2,32,36]")
    orography = value[1]
    scale = float(orography.std())
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("orography has invalid scale")
    value[1] = (orography - float(orography.mean())) / scale
    return value


class ForecastWindowDataset(Dataset):
    """Returns physical history/targets; normalization stays inside the rollout."""

    def __init__(
        self,
        climate: ClimateDataset,
        window_rows: np.ndarray,
        transform: ChannelTransform,
        history: int = 3,
        horizon: int = 2,
    ):
        rows = np.asarray(window_rows, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != history + horizon:
            raise ValueError("window rows do not match history+horizon")
        if rows.size and (rows.min() < 0 or rows.max() >= climate.data.shape[0]):
            raise ValueError("window index outside dataset")
        self.climate = climate
        self.rows = rows
        self.transform = transform
        self.history = history
        self.horizon = horizon
        self.static = torch.from_numpy(normalized_static(climate.static))

    def __len__(self) -> int:
        return self.rows.shape[0]

    def __getitem__(self, index: int):
        row = self.rows[index]
        history_indices = row[: self.history]
        target_indices = row[self.history :]
        last_input_date = self.climate.time[history_indices[-1]]
        return {
            "history": torch.from_numpy(self.climate.data[history_indices].copy()),
            "target": torch.from_numpy(self.climate.data[target_indices].copy()),
            "static": self.static,
            "doy_sincos": torch.from_numpy(doy_sincos(last_input_date)),
        }

