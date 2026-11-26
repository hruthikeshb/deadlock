from __future__ import annotations

import numpy as np

from deadlock_monsoon.baselines import CalendarDayClimatology, persistence_forecast
from deadlock_monsoon.metrics import critical_success_index, latitude_weighted_rmse


def test_zero_rmse_and_persistence_shape():
    state = np.zeros((6, 32, 36), dtype=np.float32)
    forecast = persistence_forecast(state, 10)
    assert forecast.shape == (10, 6, 32, 36)
    assert np.array_equal(latitude_weighted_rmse(forecast, forecast), np.zeros(6))


def test_csi_contingency_counts():
    prediction = np.array([0.0, 11.0, 12.0, 0.0])
    target = np.array([0.0, 13.0, 0.0, 14.0])
    result = critical_success_index(prediction, target)
    assert result["hits"] == 1
    assert result["misses"] == 1
    assert result["false_alarms"] == 1
    assert result["csi"] == 1 / 3
    assert result["false_alarm_ratio"] == 1 / 2


def test_climatology_uses_only_training_rows():
    times = np.arange(np.datetime64("2016-01-01"), np.datetime64("2018-01-01"))
    data = np.ones((times.size, 6, 2, 2), dtype=np.float32)
    mask = times < np.datetime64("2017-01-01")
    data[~mask] = 999.0
    climatology = CalendarDayClimatology.fit(data, times, mask)
    predicted = climatology.predict(np.array([np.datetime64("2017-07-15")]))
    assert np.array_equal(predicted, np.ones((1, 6, 2, 2), dtype=np.float32))

