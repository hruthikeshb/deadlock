"""Judge-facing array contract and validation.

This module contains no model logic. All engines receive the same validated,
copied physical-unit arrays, preventing scenario-specific serving behavior.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
STATE_SHAPE = (6, 32, 36)
INIT_SHAPE = (3, *STATE_SHAPE)
MAX_STEPS = 14
T2M_MIN_K = 233.15
T2M_MAX_K = 333.15


class ContractError(ValueError):
    """Raised when a client payload violates the public API contract."""


def decode_array(value: Any, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    """Decode canonical nested lists or the tolerated little-endian float32 form."""
    try:
        if isinstance(value, dict):
            if set(value) != {"b64", "shape"}:
                raise ContractError(f"{name}: encoded array needs only 'b64' and 'shape'")
            shape = tuple(int(part) for part in value["shape"])
            if shape != expected_shape:
                raise ContractError(f"{name}: shape {shape} != {expected_shape}")
            raw = base64.b64decode(value["b64"], validate=True)
            if len(raw) != int(np.prod(shape)) * 4:
                raise ContractError(f"{name}: encoded byte length does not match shape")
            array = np.frombuffer(raw, dtype="<f4").reshape(shape)
        else:
            array = np.asarray(value, dtype=np.float32)
    except ContractError:
        raise
    except (TypeError, ValueError, base64.binascii.Error) as exc:
        raise ContractError(f"{name}: cannot decode float32 array") from exc

    if array.shape != expected_shape:
        raise ContractError(f"{name}: shape {array.shape} != {expected_shape}")
    if not np.isfinite(array).all():
        raise ContractError(f"{name}: values must be finite")
    return np.array(array, dtype=np.float32, copy=True, order="C")


def validate_steps(value: Any) -> int:
    if isinstance(value, bool):
        raise ContractError("steps must be an integer from 1 through 14")
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("steps must be an integer from 1 through 14") from exc
    if steps != value or not 1 <= steps <= MAX_STEPS:
        raise ContractError("steps must be an integer from 1 through 14")
    return steps


def validate_doy_sincos(value: Any) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ContractError("doy_sincos must contain two finite numbers")
    norm = float(np.linalg.norm(array))
    if not 0.95 <= norm <= 1.05:
        raise ContractError("doy_sincos must be a sine/cosine pair on the unit circle")
    return float(array[0]), float(array[1])


def validate_rollout(value: np.ndarray, steps: int) -> np.ndarray:
    rollout = np.asarray(value, dtype=np.float32)
    expected = (steps, *STATE_SHAPE)
    if rollout.shape != expected:
        raise RuntimeError(f"engine returned shape {rollout.shape}, expected {expected}")
    if not np.isfinite(rollout).all():
        raise RuntimeError("engine returned nonfinite values")
    if np.any(rollout[:, 1] < 0):
        raise RuntimeError("engine returned negative precipitation")
    if np.any((rollout[:, 0] < T2M_MIN_K) | (rollout[:, 0] > T2M_MAX_K)):
        raise RuntimeError("engine returned temperature outside the validity interval")
    return np.array(rollout, dtype=np.float32, copy=False, order="C")

