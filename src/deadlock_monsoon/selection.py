"""Rollout-aware checkpoint selection without hiding invalid or late-lead failures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CheckpointScore:
    aggregate_normalized_rmse: float
    day1_normalized_rmse: float
    day4_normalized_rmse: float
    day8_normalized_rmse: float
    day10_normalized_rmse: float
    valid: bool
    passes_anti_blur: bool
    passes_rain_activity: bool
    passes_false_alarm: bool
    beats_persistence_day4: bool
    release_eligible: bool


def score_checkpoint(
    rmse_by_lead: dict[int, np.ndarray],
    channel_scale: np.ndarray,
    persistence_day4_rmse: np.ndarray,
    valid: bool,
    passes_anti_blur: bool = True,
    passes_rain_activity: bool = True,
    passes_false_alarm: bool = True,
) -> CheckpointScore:
    required = (1, 4, 8, 10)
    if tuple(sorted(rmse_by_lead)) != required:
        raise ValueError("checkpoint selection requires exactly days 1, 4, 8, and 10")
    scale = np.asarray(channel_scale, dtype=np.float64)
    reference = np.asarray(persistence_day4_rmse, dtype=np.float64)
    if scale.shape != (6,) or reference.shape != (6,) or np.any(scale <= 0):
        raise ValueError("channel/reference arrays must contain six valid values")
    normalized = {}
    for lead in required:
        value = np.asarray(rmse_by_lead[lead], dtype=np.float64)
        if value.shape != (6,) or not np.isfinite(value).all():
            valid = False
            normalized[lead] = float("inf")
        else:
            normalized[lead] = float(np.mean(value / scale))
    day4 = np.asarray(rmse_by_lead[4], dtype=np.float64)
    eligible = bool(valid and passes_anti_blur and passes_rain_activity and passes_false_alarm)
    beats_persistence = bool(eligible and np.mean(day4 / scale) < np.mean(reference / scale))
    aggregate = float(np.mean([normalized[lead] for lead in required])) if eligible else float("inf")
    return CheckpointScore(
        aggregate_normalized_rmse=aggregate,
        day1_normalized_rmse=normalized[1],
        day4_normalized_rmse=normalized[4],
        day8_normalized_rmse=normalized[8],
        day10_normalized_rmse=normalized[10],
        valid=eligible,
        passes_anti_blur=bool(passes_anti_blur),
        passes_rain_activity=bool(passes_rain_activity),
        passes_false_alarm=bool(passes_false_alarm),
        beats_persistence_day4=beats_persistence,
        release_eligible=bool(eligible and beats_persistence),
    )


def is_better(candidate: CheckpointScore, incumbent: CheckpointScore | None) -> bool:
    if not candidate.release_eligible:
        return False
    if incumbent is None or not incumbent.release_eligible:
        return True
    return candidate.aggregate_normalized_rmse < incumbent.aggregate_normalized_rmse

