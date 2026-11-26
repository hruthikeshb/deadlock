"""Forecast engine protocol and the always-runnable P0 persistence baseline."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ForecastEngine(Protocol):
    name: str

    def simulate(
        self,
        init_states: np.ndarray,
        doy_sincos: tuple[float, float],
        steps: int,
        perturbation: np.ndarray | None,
    ) -> np.ndarray: ...


class PersistenceEngine:
    """Deterministic contract baseline; never a claimed learned forecast."""

    name = "persistence-p0"

    def simulate(
        self,
        init_states: np.ndarray,
        doy_sincos: tuple[float, float],
        steps: int,
        perturbation: np.ndarray | None,
    ) -> np.ndarray:
        del doy_sincos
        state = np.array(init_states[-1], dtype=np.float32, copy=True)
        if perturbation is not None:
            state += perturbation
        return np.repeat(state[None, ...], steps, axis=0)

