from __future__ import annotations

import numpy as np

from deadlock_monsoon.dataset import EXPECTED_LAT, EXPECTED_LON
from deadlock_monsoon.geography import geography_features


def test_static_control_has_land_sea_and_normalized_orography():
    static = np.zeros((2, 32, 36), dtype=np.float32)
    static[0, :, :18] = 1.0
    static[1] = np.arange(32 * 36, dtype=np.float32).reshape(32, 36)
    features = geography_features(static, EXPECTED_LAT, EXPECTED_LON, mode="static")
    assert features.shape == (2, 32, 36)
    assert np.array_equal(features[0], static[0])
    assert abs(float(features[1].mean())) < 1e-5
    assert np.isclose(float(features[1].std()), 1.0, atol=1e-5)


def test_metric_features_encode_coordinates_and_latitude_dependent_dx():
    static = np.zeros((2, 32, 36), dtype=np.float32)
    static[1] = np.arange(32 * 36, dtype=np.float32).reshape(32, 36)
    features = geography_features(static, EXPECTED_LAT, EXPECTED_LON, mode="metric")
    assert features.shape == (8, 32, 36)
    assert np.allclose(features[2, :, 0], np.sin(np.deg2rad(EXPECTED_LAT)))
    assert np.allclose(features[4, 0], np.sin(np.deg2rad(EXPECTED_LON)))
    assert features[6, 0, 0] > features[6, -1, 0]
    assert np.allclose(features[7], 1.0)

