"""Deterministic static and metric-aware geography feature construction."""

from __future__ import annotations

import numpy as np


def geography_features(
    static: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mode: str = "static",
) -> np.ndarray:
    fields = np.asarray(static, dtype=np.float32)
    latitude = np.asarray(lat, dtype=np.float64)
    longitude = np.asarray(lon, dtype=np.float64)
    if fields.shape != (2, 32, 36):
        raise ValueError("static fields must have shape [2,32,36]")
    if latitude.shape != (32,) or longitude.shape != (36,):
        raise ValueError("latitude/longitude shapes do not match the regional grid")
    if mode not in {"static", "metric"}:
        raise ValueError("geography mode must be static or metric")

    land_sea = fields[0].copy()
    orography = fields[1]
    orography_scale = float(orography.std())
    if orography_scale <= 0 or not np.isfinite(orography_scale):
        raise ValueError("orography has invalid scale")
    normalized_orography = (orography - float(orography.mean())) / orography_scale
    base = [land_sea, normalized_orography]
    if mode == "static":
        return np.stack(base).astype(np.float32)

    lat_radians = np.deg2rad(latitude)
    lon_radians = np.deg2rad(longitude)
    sin_lat = np.broadcast_to(np.sin(lat_radians)[:, None], (32, 36))
    cos_lat = np.broadcast_to(np.cos(lat_radians)[:, None], (32, 36))
    sin_lon = np.broadcast_to(np.sin(lon_radians)[None, :], (32, 36))
    cos_lon = np.broadcast_to(np.cos(lon_radians)[None, :], (32, 36))
    # For a regular 1-degree grid, zonal physical spacing scales with cos(latitude).
    dx_ratio = cos_lat / cos_lat.mean()
    dy_ratio = np.ones((32, 36), dtype=np.float64)
    return np.stack(base + [sin_lat, cos_lat, sin_lon, cos_lon, dx_ratio, dy_ratio]).astype(
        np.float32
    )

