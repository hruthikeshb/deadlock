"""Leakage-resistant chronological windows and training-partition transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import STATE_SHAPE


@dataclass(frozen=True)
class SplitSpec:
    train_end: np.datetime64 = np.datetime64("2017-12-31")
    validation_end: np.datetime64 = np.datetime64("2019-12-31")
    test_end: np.datetime64 = np.datetime64("2021-12-31")


@dataclass(frozen=True)
class ChannelTransform:
    """Per-channel affine transform fitted exclusively on declared training rows."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray, train_mask: np.ndarray) -> "ChannelTransform":
        values = np.asarray(data, dtype=np.float64)
        mask = np.asarray(train_mask, dtype=bool)
        if values.ndim != 4 or values.shape[1:] != STATE_SHAPE:
            raise ValueError(f"data shape must be [time,{STATE_SHAPE}], got {values.shape}")
        if mask.shape != (values.shape[0],) or not mask.any():
            raise ValueError("train_mask must select at least one time row")
        train = values[mask]
        if not np.isfinite(train).all():
            raise ValueError("training values must be finite")
        mean = train.mean(axis=(0, 2, 3), dtype=np.float64)
        scale = train.std(axis=(0, 2, 3), dtype=np.float64)
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError("every channel needs positive finite training variance")
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        channel_axis = array.ndim - 3
        if channel_axis < 0 or array.shape[channel_axis] != STATE_SHAPE[0]:
            raise ValueError("expected channel axis immediately before latitude/longitude")
        shape = [1] * array.ndim
        shape[channel_axis] = STATE_SHAPE[0]
        return (array - self.mean.reshape(shape)) / self.scale.reshape(shape)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        channel_axis = array.ndim - 3
        if channel_axis < 0 or array.shape[channel_axis] != STATE_SHAPE[0]:
            raise ValueError("expected channel axis immediately before latitude/longitude")
        shape = [1] * array.ndim
        shape[channel_axis] = STATE_SHAPE[0]
        return array * self.scale.reshape(shape) + self.mean.reshape(shape)


def chronological_masks(times: np.ndarray, split: SplitSpec = SplitSpec()):
    dates = np.asarray(times, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if np.any(np.diff(dates).astype("timedelta64[D]") != np.timedelta64(1, "D")):
        raise ValueError("times must be strictly daily and contiguous")
    train = dates <= split.train_end
    validation = (dates > split.train_end) & (dates <= split.validation_end)
    test = (dates > split.validation_end) & (dates <= split.test_end)
    return {"train": train, "validation": validation, "test": test}


def window_indices(mask: np.ndarray, history: int = 3, horizon: int = 14) -> np.ndarray:
    """Return complete input+target index rows wholly contained in one partition."""
    membership = np.asarray(mask, dtype=bool)
    if membership.ndim != 1 or history < 1 or not 1 <= horizon <= 14:
        raise ValueError("invalid mask, history, or horizon")
    width = history + horizon
    rows = [
        np.arange(start, start + width, dtype=np.int64)
        for start in range(0, membership.size - width + 1)
        if membership[start : start + width].all()
    ]
    return np.stack(rows) if rows else np.empty((0, width), dtype=np.int64)

