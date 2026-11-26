"""Leakage-free persistence and calendar-day climatology baselines."""

from __future__ import annotations

import numpy as np


def persistence_forecast(last_state: np.ndarray, steps: int) -> np.ndarray:
    state = np.asarray(last_state, dtype=np.float32)
    if state.shape != (6, 32, 36) or not 1 <= steps <= 14:
        raise ValueError("invalid persistence state or horizon")
    return np.repeat(state[None], steps, axis=0)


def month_day_keys(times: np.ndarray) -> np.ndarray:
    dates = np.asarray(times, dtype="datetime64[D]")
    return np.array([np.datetime_as_string(date, unit="D")[5:] for date in dates])


class CalendarDayClimatology:
    def __init__(self, values: dict[str, np.ndarray]):
        self._values = values

    @classmethod
    def fit(cls, data: np.ndarray, times: np.ndarray, train_mask: np.ndarray):
        array = np.asarray(data, dtype=np.float32)
        dates = np.asarray(times, dtype="datetime64[D]")
        mask = np.asarray(train_mask, dtype=bool)
        if array.shape[0] != dates.size or mask.shape != (dates.size,):
            raise ValueError("data, times, and train mask lengths differ")
        keys = month_day_keys(dates)
        values = {
            key: array[mask & (keys == key)].mean(axis=0, dtype=np.float64).astype(np.float32)
            for key in np.unique(keys[mask])
        }
        if len(values) < 365:
            raise ValueError("training partition does not cover a complete annual cycle")
        return cls(values)

    def predict(self, forecast_times: np.ndarray) -> np.ndarray:
        keys = month_day_keys(forecast_times)
        missing = sorted(set(keys) - self._values.keys())
        if missing:
            raise ValueError(f"climatology lacks calendar days: {missing}")
        return np.stack([self._values[key] for key in keys]).astype(np.float32)

