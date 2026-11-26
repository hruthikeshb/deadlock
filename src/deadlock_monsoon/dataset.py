"""Verified organizer NetCDF reader used only by training and evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import CHANNELS, STATE_SHAPE

EXPECTED_DATASET_SHA256 = "f37bdbd0bd44e6fe951adf36c9a08620075f9d17de391b04c97d61f06797903b"
STATIC_CHANNELS = ("land_sea_mask", "orography_m")
EXPECTED_LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
EXPECTED_LON = np.arange(65.5, 100.5 + 1e-9, 1.0)
EXPECTED_UNITS = "t2m K | tp mm/day | u10 m/s | v10 m/s | msl Pa | tcwv kg/m2"


@dataclass(frozen=True)
class ClimateDataset:
    data: np.ndarray
    static: np.ndarray
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_dataset(
    path: str | Path,
    expected_sha256: str = EXPECTED_DATASET_SHA256,
) -> ClimateDataset:
    """Load the official dataset and fail closed on integrity or schema drift."""
    source = Path(path)
    actual_hash = sha256_file(source)
    if actual_hash != expected_sha256:
        raise ValueError(f"dataset SHA-256 mismatch: {actual_hash}")

    try:
        import xarray as xr
    except ImportError as exc:  # training-only dependency
        raise RuntimeError("xarray is required to read the training dataset") from exc

    with xr.open_dataset(source) as dataset:
        required_dims = {"time": 6209, "channel": 6, "lat": 32, "lon": 36, "static_channel": 2}
        if dict(dataset.sizes) != required_dims:
            raise ValueError(f"unexpected dimensions: {dict(dataset.sizes)}")
        if tuple(dataset["data"].dims) != ("time", "channel", "lat", "lon"):
            raise ValueError("data dimensions/order do not match the contract")
        if tuple(dataset["static"].dims) != ("static_channel", "lat", "lon"):
            raise ValueError("static dimensions/order do not match the contract")
        if tuple(str(x) for x in dataset.channel.values) != CHANNELS:
            raise ValueError("dynamic channel order mismatch")
        if tuple(str(x) for x in dataset.static_channel.values) != STATIC_CHANNELS:
            raise ValueError("static channel order mismatch")
        if not np.array_equal(dataset.lat.values, EXPECTED_LAT):
            raise ValueError("latitude coordinates mismatch")
        if not np.array_equal(dataset.lon.values, EXPECTED_LON):
            raise ValueError("longitude coordinates mismatch")
        if dataset["data"].attrs.get("units") != EXPECTED_UNITS:
            raise ValueError("dynamic units metadata mismatch")
        data = dataset["data"].values.astype(np.float32, copy=True)
        static = dataset["static"].values.astype(np.float32, copy=True)
        time = dataset.time.values.astype("datetime64[D]")

    if data.shape[1:] != STATE_SHAPE or static.shape != (2, 32, 36):
        raise ValueError("loaded arrays have unexpected shapes")
    if not np.isfinite(data).all() or not np.isfinite(static).all():
        raise ValueError("dataset contains nonfinite values")
    if np.any(data[:, 1] < 0):
        raise ValueError("dataset contains negative precipitation")
    return ClimateDataset(data, static, time, EXPECTED_LAT.copy(), EXPECTED_LON.copy())

