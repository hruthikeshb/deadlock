"""Scientific diagnostics for rollout drift and perturbation response."""

from __future__ import annotations

import numpy as np


def normalized_error_by_lead(
    prediction: np.ndarray,
    target: np.ndarray,
    channel_scale: np.ndarray,
) -> np.ndarray:
    """Return normalized RMSE [lead, channel] without averaging away degradation."""
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    scale = np.asarray(channel_scale, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 5 or pred.shape[2:] != (6, 32, 36):
        raise ValueError("rollouts must have shape [batch,lead,6,32,36]")
    if scale.shape != (6,) or np.any(scale <= 0):
        raise ValueError("channel scale must contain six positive values")
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 3, 4)))
    return rmse / scale[None, :]


def cross_channel_error_correlation(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return [lead,6,6] correlations of flattened channel errors."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if error.ndim != 5 or error.shape[2:] != (6, 32, 36):
        raise ValueError("rollouts must have shape [batch,lead,6,32,36]")
    matrices = []
    for lead in range(error.shape[1]):
        flattened = error[:, lead].transpose(1, 0, 2, 3).reshape(6, -1)
        matrices.append(np.corrcoef(flattened))
    return np.stack(matrices)


def response_energy(
    base_rollout: np.ndarray,
    perturbed_rollout: np.ndarray,
    channel_scale: np.ndarray,
) -> np.ndarray:
    """Squared standardized response energy [lead,lat,lon]."""
    base = np.asarray(base_rollout, dtype=np.float64)
    shocked = np.asarray(perturbed_rollout, dtype=np.float64)
    scale = np.asarray(channel_scale, dtype=np.float64)
    if base.shape != shocked.shape or base.ndim != 4 or base.shape[1:] != (6, 32, 36):
        raise ValueError("base and perturbed rollouts must have shape [lead,6,32,36]")
    if scale.shape != (6,) or np.any(scale <= 0):
        raise ValueError("channel scale must contain six positive values")
    delta = (shocked - base) / scale[None, :, None, None]
    return np.sum(delta**2, axis=1)


def dilate_mask(mask: np.ndarray, cells: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.shape != (32, 36) or cells < 0:
        raise ValueError("mask/cell radius invalid")
    result = value.copy()
    for _ in range(cells):
        previous = result.copy()
        result[1:] |= previous[:-1]
        result[:-1] |= previous[1:]
        result[:, 1:] |= previous[:, :-1]
        result[:, :-1] |= previous[:, 1:]
    return result


def perturbation_report(
    base_rollout: np.ndarray,
    perturbed_rollout: np.ndarray,
    channel_scale: np.ndarray,
    source_mask: np.ndarray,
    near_cells: int = 3,
) -> dict:
    base = np.asarray(base_rollout, dtype=np.float64)
    shocked = np.asarray(perturbed_rollout, dtype=np.float64)
    delta = shocked - base
    finite = bool(np.isfinite(delta).all())
    if not finite:
        return {"finite": False, "nonzero": False, "stable": False}
    energy = response_energy(base, shocked, channel_scale)
    near = dilate_mask(source_mask, near_cells)
    totals = energy.sum(axis=(1, 2))
    near_fraction = np.divide(
        energy[:, near].sum(axis=1),
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    y, x = np.indices((32, 36))
    centroids = []
    radii = []
    for lead_energy, total in zip(energy, totals):
        if total <= 0:
            centroids.append([float("nan"), float("nan")])
            radii.append(float("nan"))
            continue
        cy = float(np.sum(lead_energy * y) / total)
        cx = float(np.sum(lead_energy * x) / total)
        radius = float(np.sqrt(np.sum(lead_energy * ((y - cy) ** 2 + (x - cx) ** 2)) / total))
        centroids.append([cy, cx])
        radii.append(radius)
    scale = np.asarray(channel_scale, dtype=np.float64)
    standardized_max = float(np.max(np.abs(delta) / scale[None, :, None, None]))
    return {
        "finite": True,
        "nonzero": bool(standardized_max >= 1e-3),
        "stable": bool(standardized_max <= 5.0),
        "max_response_std": standardized_max,
        "near_fraction": near_fraction.tolist(),
        "outside_energy": (totals - energy[:, near].sum(axis=1)).tolist(),
        "centroid_cells": centroids,
        "radius_cells": radii,
    }

