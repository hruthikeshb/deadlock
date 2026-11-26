from __future__ import annotations

import numpy as np

from deadlock_monsoon.selection import is_better, score_checkpoint


def test_selection_requires_all_long_horizon_checkpoints():
    rmse = {lead: np.full(6, lead / 10.0) for lead in (1, 4, 8, 10)}
    score = score_checkpoint(rmse, np.ones(6), np.ones(6), valid=True)
    assert score.valid
    assert score.beats_persistence_day4
    assert score.release_eligible
    assert score.day10_normalized_rmse > score.day1_normalized_rmse


def test_invalid_candidate_never_replaces_valid_checkpoint():
    rmse = {lead: np.ones(6) for lead in (1, 4, 8, 10)}
    valid = score_checkpoint(rmse, np.ones(6), np.full(6, 2.0), valid=True)
    invalid = score_checkpoint(rmse, np.ones(6), np.full(6, 2.0), valid=False)
    assert is_better(valid, None)
    assert not is_better(invalid, valid)


def test_blurred_or_rainless_candidate_is_ineligible_even_when_rmse_is_low():
    rmse = {lead: np.full(6, 0.1) for lead in (1, 4, 8, 10)}
    blurred = score_checkpoint(
        rmse, np.ones(6), np.ones(6), valid=True,
        passes_anti_blur=False, passes_rain_activity=True,
    )
    rainless = score_checkpoint(
        rmse, np.ones(6), np.ones(6), valid=True,
        passes_anti_blur=True, passes_rain_activity=False,
    )
    assert not blurred.valid and not rainless.valid
    assert not is_better(blurred, None)
    assert not is_better(rainless, None)


def test_scientifically_valid_candidate_still_needs_day4_persistence_gate():
    rmse = {lead: np.ones(6) for lead in (1, 4, 8, 10)}
    score = score_checkpoint(rmse, np.ones(6), np.full(6, 0.5), valid=True)
    assert score.valid
    assert not score.beats_persistence_day4
    assert not score.release_eligible
    assert not is_better(score, None)

