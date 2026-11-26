from __future__ import annotations

import numpy as np
import pytest

from deadlock_monsoon.contract import ContractError, INIT_SHAPE, STATE_SHAPE, validate_steps
from deadlock_monsoon.engine import PersistenceEngine
from deadlock_monsoon.service import run_request


def valid_state() -> np.ndarray:
    state = np.zeros(INIT_SHAPE, dtype=np.float32)
    state[:, 0] = 298.0
    state[:, 1] = 2.0
    state[:, 4] = 100_000.0
    state[:, 5] = 30.0
    return state


def payload(steps: int = 10, perturbation=None):
    return {
        "init_states": valid_state().tolist(),
        "doy_sincos": [0.0, 1.0],
        "steps": steps,
        "perturbation": perturbation,
    }


@pytest.mark.parametrize("steps", range(1, 15))
def test_every_supported_horizon(steps: int):
    rollout = run_request(PersistenceEngine(), payload(steps))
    assert rollout.shape == (steps, *STATE_SHAPE)
    assert np.isfinite(rollout).all()


@pytest.mark.parametrize("steps", [0, 15, True, 1.5, "ten", None])
def test_rejects_invalid_horizons(steps):
    with pytest.raises(ContractError):
        validate_steps(steps)


def test_perturbation_is_applied_once_without_mutating_input():
    init = valid_state()
    original = init.copy()
    perturbation = np.zeros(STATE_SHAPE, dtype=np.float32)
    perturbation[0, 10:12, 10:12] = 1.5
    body = payload(4, perturbation.tolist())
    body["init_states"] = init.tolist()
    rollout = run_request(PersistenceEngine(), body)
    assert np.array_equal(init, original)
    expected = original[-1] + perturbation
    assert np.array_equal(rollout[0], expected)
    assert np.array_equal(rollout[3], expected)


def test_null_and_zero_perturbations_are_identical_and_deterministic():
    engine = PersistenceEngine()
    base = run_request(engine, payload(8, None))
    zero = run_request(engine, payload(8, np.zeros(STATE_SHAPE).tolist()))
    repeated = run_request(engine, payload(8, None))
    assert np.array_equal(base, zero)
    assert np.array_equal(base, repeated)


def test_rejects_nonfinite_input():
    body = payload(1)
    body["init_states"][0][0][0][0] = float("nan")
    with pytest.raises(ContractError, match="finite"):
        run_request(PersistenceEngine(), body)


def test_rejects_wrong_shape():
    body = payload(1)
    body["init_states"] = body["init_states"][:2]
    with pytest.raises(ContractError, match="shape"):
        run_request(PersistenceEngine(), body)

