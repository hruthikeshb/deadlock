from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deadlock_monsoon.data import ChannelTransform
from deadlock_monsoon.dataset import ClimateDataset, EXPECTED_LAT, EXPECTED_LON
from deadlock_monsoon.train_data import ForecastWindowDataset, doy_sincos, normalized_static


def synthetic_climate() -> ClimateDataset:
    rng = np.random.default_rng(4)
    data = rng.normal(size=(8, 6, 32, 36)).astype(np.float32)
    data[:, 0] += 298.0
    data[:, 1] = np.abs(data[:, 1])
    data[:, 4] += 100_000.0
    static = rng.normal(size=(2, 32, 36)).astype(np.float32)
    static[0] = np.clip(static[0], 0, 1)
    times = np.datetime64("2016-01-01") + np.arange(8).astype("timedelta64[D]")
    return ClimateDataset(data, static, times, EXPECTED_LAT, EXPECTED_LON)


def test_window_dataset_alignment_and_shapes():
    climate = synthetic_climate()
    transform = ChannelTransform.fit(climate.data, np.ones(8, dtype=bool))
    rows = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])
    dataset = ForecastWindowDataset(climate, rows, transform, history=3, horizon=2)
    item = dataset[0]
    assert item["history"].shape == (3, 6, 32, 36)
    assert item["target"].shape == (2, 6, 32, 36)
    assert item["static"].shape == (2, 32, 36)
    assert np.array_equal(item["history"].numpy(), climate.data[:3])
    assert np.array_equal(item["target"].numpy(), climate.data[3:5])


def test_static_and_seasonal_features_are_well_formed():
    climate = synthetic_climate()
    static = normalized_static(climate.static)
    assert abs(float(static[1].mean())) < 1e-5
    assert np.isclose(float(static[1].std()), 1.0, atol=1e-5)
    pair = doy_sincos(np.datetime64("2016-07-01"))
    assert np.isclose(np.linalg.norm(pair), 1.0, atol=1e-6)

