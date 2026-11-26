from __future__ import annotations

import numpy as np
import pytest

from deadlock_monsoon.contract import STATE_SHAPE
from deadlock_monsoon.data import ChannelTransform, SplitSpec, chronological_masks, window_indices


def daily(start: str, count: int):
    origin = np.datetime64(start, "D")
    return origin + np.arange(count).astype("timedelta64[D]")


def test_windows_never_cross_partition_boundaries():
    times = daily("2017-12-20", 30)
    masks = chronological_masks(times, SplitSpec())
    for mask in masks.values():
        rows = window_indices(mask, history=3, horizon=4)
        assert all(mask[row].all() for row in rows)


def test_future_values_cannot_change_training_statistics():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(4, *STATE_SHAPE)).astype(np.float32)
    train_mask = np.array([True, True, False, False])
    first = ChannelTransform.fit(data, train_mask)
    data[~train_mask] = 1e9
    second = ChannelTransform.fit(data, train_mask)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.scale, second.scale)


def test_transform_round_trip():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(5, *STATE_SHAPE)).astype(np.float32)
    transform = ChannelTransform.fit(data, np.array([True, True, True, False, False]))
    restored = transform.denormalize(transform.normalize(data))
    assert np.allclose(restored, data, rtol=1e-5, atol=1e-6)


def test_noncontiguous_dates_fail_closed():
    times = daily("2017-01-01", 5)
    times[3] += np.timedelta64(1, "D")
    with pytest.raises(ValueError, match="contiguous"):
        chronological_masks(times)

