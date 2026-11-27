"""
GeoFeatures
===========
Builds fixed static + coordinate channels that are concatenated with the
dynamic history before being fed to the model.

Channels produced (all float32, shape [C, 32, 36]):
    0  : LSM             (fractional land-sea mask, 0–1)
    1  : orography_norm  (orography normalized to [0,1] by training max)
    2  : lat_norm        (latitude  normalized to [-1, 1])
    3  : lon_norm        (longitude normalized to [-1, 1])
    4  : cos_lat         (cos of latitude in radians — metric-weight proxy)
    5  : dx_norm         (physical E-W cell width in km, normalized)

Total: N_GEO_CHANNELS = 6
"""

from __future__ import annotations

import numpy as np

N_GEO_CHANNELS = 6

# India-domain coordinate bounds (from dataset_test.py)
LAT_MIN, LAT_MAX = 6.5, 37.5
LON_MIN, LON_MAX = 65.5, 100.5
EARTH_RADIUS_KM  = 6371.0


def build_geo_features(
    lsm: np.ndarray,        # [32, 36] fractional land-sea mask
    orography: np.ndarray,  # [32, 36] orography in metres
    lats: np.ndarray,       # [32]  latitude centres in degrees
    lons: np.ndarray,       # [36]  longitude centres in degrees
) -> np.ndarray:
    """
    Return geo-feature tensor of shape [N_GEO_CHANNELS, 32, 36].
    All values are float32 and finite.
    """
    H, W = lsm.shape

    # --- LSM (keep as-is, already in [0,1]) ---
    lsm_ch = lsm.astype("float32")

    # --- Orography: normalize to [0,1] ---
    oro_min = orography.min()
    oro_max = max(orography.max(), 1.0)      # guard against flat grid
    oro_ch = ((orography - oro_min) / (oro_max - oro_min)).astype("float32")

    # --- Latitude grid [H, W] normalized to [-1, 1] ---
    lat_1d = ((lats - LAT_MIN) / (LAT_MAX - LAT_MIN) * 2 - 1).astype("float32")
    lat_ch = np.broadcast_to(lat_1d[:, None], (H, W)).copy()

    # --- Longitude grid [H, W] normalized to [-1, 1] ---
    lon_1d = ((lons - LON_MIN) / (LON_MAX - LON_MIN) * 2 - 1).astype("float32")
    lon_ch = np.broadcast_to(lon_1d[None, :], (H, W)).copy()

    # --- cos(lat) in radians — proxy for E-W metric scale ---
    cos_lat_1d = np.cos(np.deg2rad(lats)).astype("float32")
    cos_lat_ch = np.broadcast_to(cos_lat_1d[:, None], (H, W)).copy()

    # --- Physical E-W cell width in km (1° lon at each lat) ---
    # dx = R * cos(lat) * Δlon_rad
    delta_lon_rad = np.deg2rad(1.0)           # 1° spacing
    dx_km_1d = (EARTH_RADIUS_KM * np.cos(np.deg2rad(lats)) * delta_lon_rad).astype("float32")
    dx_norm_1d = (dx_km_1d / dx_km_1d.max()).astype("float32")
    dx_ch = np.broadcast_to(dx_norm_1d[:, None], (H, W)).copy()

    geo = np.stack([lsm_ch, oro_ch, lat_ch, lon_ch, cos_lat_ch, dx_ch], axis=0)
    assert geo.shape == (N_GEO_CHANNELS, H, W)
    assert np.isfinite(geo).all()
    return geo
