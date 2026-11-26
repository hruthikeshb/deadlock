from __future__ import annotations

import numpy as np

from deadlock_monsoon.checkpoint import export_grast_bundle
from deadlock_monsoon.learned_engine import GRASTEngine
from deadlock_monsoon.model import GRAST, GRASTConfig


def physical_history() -> np.ndarray:
    value = np.zeros((3, 6, 32, 36), dtype=np.float32)
    value[:, 0] = 298.0
    value[:, 1] = 2.0
    value[:, 4] = 100_000.0
    value[:, 5] = 30.0
    return value


def test_packaged_engine_is_deterministic_physical_and_does_not_mutate_input(tmp_path):
    model = GRAST(GRASTConfig(width=16, blocks=2, rain_mode="occurrence_intensity"))
    export_grast_bundle(
        tmp_path,
        model,
        role="primary",
        normalization_mean=np.zeros(6, dtype=np.float32),
        normalization_scale=np.ones(6, dtype=np.float32),
        split={"train_end": "2017-12-31"},
        training={"seed": 7},
        code_commit="testcommit",
        static_features=np.zeros((2, 32, 36), dtype=np.float32),
    )
    engine = GRASTEngine(tmp_path, threads=1)
    history = physical_history()
    original = history.copy()
    first = engine.simulate(history, (0.0, 1.0), 2, None)
    second = engine.simulate(history, (0.0, 1.0), 2, None)
    assert np.array_equal(history, original)
    assert np.array_equal(first, second)
    assert first.shape == (2, 6, 32, 36)
    assert np.isfinite(first).all()
    assert np.all(first[:, 1] >= 0)
    assert np.all((first[:, 0] >= 233.15) & (first[:, 0] <= 333.15))
