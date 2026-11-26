from __future__ import annotations

import numpy as np

from deadlock_monsoon.diagnostics import (
    cross_channel_error_correlation,
    normalized_error_by_lead,
    perturbation_report,
    response_energy,
)


def test_normalized_error_is_reported_per_lead_and_channel():
    target = np.zeros((2, 4, 6, 32, 36))
    prediction = target.copy()
    prediction[:, 3, 2] = 2.0
    result = normalized_error_by_lead(prediction, target, np.ones(6))
    assert result.shape == (4, 6)
    assert result[0, 2] == 0
    assert result[3, 2] == 2


def test_cross_channel_error_correlation_shape():
    rng = np.random.default_rng(5)
    target = np.zeros((3, 2, 6, 32, 36))
    prediction = rng.normal(size=target.shape)
    correlation = cross_channel_error_correlation(prediction, target)
    assert correlation.shape == (2, 6, 6)
    assert np.allclose(np.diagonal(correlation, axis1=1, axis2=2), 1.0)


def test_local_perturbation_report_detects_locality_and_spread():
    base = np.zeros((4, 6, 32, 36))
    shocked = base.copy()
    source = np.zeros((32, 36), dtype=bool)
    source[12:15, 12:15] = True
    shocked[0, 0, 12:15, 12:15] = 1.0
    shocked[1, 0, 11:16, 11:16] = 0.7
    shocked[2, 0, 10:17, 10:17] = 0.5
    shocked[3, 0, 9:18, 9:18] = 0.3
    energy = response_energy(base, shocked, np.ones(6))
    assert energy.shape == (4, 32, 36)
    report = perturbation_report(base, shocked, np.ones(6), source)
    assert report["finite"] and report["nonzero"] and report["stable"]
    assert report["near_fraction"][0] == 1.0
    assert report["radius_cells"][-1] > report["radius_cells"][0]


def test_nonfinite_perturbation_fails_closed():
    base = np.zeros((2, 6, 32, 36))
    shocked = base.copy()
    shocked[0, 0, 0, 0] = np.nan
    report = perturbation_report(base, shocked, np.ones(6), np.zeros((32, 36), dtype=bool))
    assert report == {"finite": False, "nonzero": False, "stable": False}

