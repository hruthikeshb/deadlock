"""Official-parity baseline metrics with explicit channel-level outputs."""

from __future__ import annotations

import numpy as np

LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)


def latitude_weighted_rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape or pred.shape[-3:] != (6, 32, 36):
        raise ValueError("prediction and target must end in [6,32,36] with identical shapes")
    weights = np.cos(np.deg2rad(LAT))
    weights /= weights.mean()
    squared = (pred - true) ** 2 * weights[None, :, None]
    return np.sqrt(squared.mean(axis=(-2, -1)).reshape(-1, 6).mean(axis=0))


def critical_success_index(
    predicted_rain: np.ndarray,
    target_rain: np.ndarray,
    threshold: float = 10.0,
) -> dict[str, float | int]:
    predicted = np.asarray(predicted_rain) > threshold
    target = np.asarray(target_rain) > threshold
    if predicted.shape != target.shape:
        raise ValueError("predicted and target rain shapes differ")
    hits = int(np.sum(predicted & target))
    misses = int(np.sum(~predicted & target))
    false_alarms = int(np.sum(predicted & ~target))
    denominator = hits + misses + false_alarms
    csi = float("nan") if denominator == 0 else hits / denominator
    precision = float("nan") if hits + false_alarms == 0 else hits / (hits + false_alarms)
    recall = float("nan") if hits + misses == 0 else hits / (hits + misses)
    false_alarm_ratio = (
        float("nan") if hits + false_alarms == 0 else false_alarms / (hits + false_alarms)
    )
    return {
        "csi": csi,
        "precision": precision,
        "recall": recall,
        "false_alarm_ratio": false_alarm_ratio,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def spatial_variance_ratio(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1, 6, 32 * 36)
    true = np.asarray(target, dtype=np.float64).reshape(-1, 6, 32 * 36)
    numerator = pred.var(axis=-1).mean(axis=0)
    denominator = true.var(axis=-1).mean(axis=0)
    return np.where(denominator > 0, numerator / denominator, np.nan)

