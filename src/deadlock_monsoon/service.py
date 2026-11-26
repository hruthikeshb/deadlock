"""Shared request-to-engine adapter used by HTTP and direct tests."""

from __future__ import annotations

from typing import Any

from .contract import (
    INIT_SHAPE,
    STATE_SHAPE,
    decode_array,
    validate_doy_sincos,
    validate_rollout,
    validate_steps,
)
from .engine import ForecastEngine


def run_request(engine: ForecastEngine, body: dict[str, Any]):
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    init_states = decode_array(body.get("init_states"), INIT_SHAPE, "init_states")
    perturbation_value = body.get("perturbation")
    perturbation = (
        None
        if perturbation_value is None
        else decode_array(perturbation_value, STATE_SHAPE, "perturbation")
    )
    doy_sincos = validate_doy_sincos(body.get("doy_sincos"))
    steps = validate_steps(body.get("steps"))
    rollout = engine.simulate(init_states, doy_sincos, steps, perturbation)
    return validate_rollout(rollout, steps)

