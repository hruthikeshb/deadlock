"""
DatasetReader + Transform + WindowIndex
=======================================
Single module for all data-layer concerns.

Usage
-----
    from src.data.dataset import DatasetReader, Transform, WindowIndex

Channel order (from stats.json / organizer contract):
    0: t2m   [K]
    1: tp    [mm/day]
    2: u10   [m/s]
    3: v10   [m/s]
    4: msl   [Pa]
    5: tcwv  [kg/m²]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
NC_PATH = DATA_DIR / "india_climate_train.nc"
STATS_PATH = DATA_DIR / "stats.json"
SPLITS_PATH = ROOT / "training" / "splits.json"

CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
N_CHANNELS = 6
GRID_H = 32
GRID_W = 36

# ---------------------------------------------------------------------------
# DatasetReader
# ---------------------------------------------------------------------------

class DatasetReader:
    """Load and validate the NetCDF dataset once; expose as numpy arrays."""

    def __init__(self, nc_path: Path = NC_PATH) -> None:
        from src.data.hdf5_reader import load_nc_arrays
        arrays = load_nc_arrays(nc_path)

        raw = arrays["data"].astype("float32")              # [T,6,32,36]
        assert raw.shape[1:] == (N_CHANNELS, GRID_H, GRID_W)
        assert np.isfinite(raw).all(), "NaN/Inf in data"
        assert raw[:, 1].min() >= 0, "negative precipitation"

        static = arrays["static"].astype("float32")         # [2,32,36]
        assert static.shape == (2, GRID_H, GRID_W)

        self.data: np.ndarray = raw
        self.static: np.ndarray = static
        self.lats: np.ndarray = arrays["lat"].astype("float32")   # [32]
        self.lons: np.ndarray = arrays["lon"].astype("float32")   # [36]

        # Build datetime64[D] axis from xarray (metadata only — no data read)
        import xarray as xr
        ds_meta = xr.open_dataset(nc_path, engine="h5netcdf")
        self._dates: np.ndarray = ds_meta["time"].values.astype("datetime64[D]")
        ds_meta.close()
        assert len(self._dates) == raw.shape[0]

    def split_indices(self, split: str) -> np.ndarray:
        """Return integer day-indices for the requested split."""
        with open(SPLITS_PATH, encoding="utf-8") as f:
            start, end = json.load(f)[split]
        s = np.datetime64(start, "D")
        e = np.datetime64(end, "D")
        idx = np.where((self._dates >= s) & (self._dates <= e))[0]
        assert len(idx) > 0, f"no days in split '{split}'"
        return idx

    def doy_sincos(self, day_idx: int) -> np.ndarray:
        """Return [sin(2π·doy/365), cos(2π·doy/365)] for a day index."""
        t = self._dates[day_idx]
        yr_start = t.astype("datetime64[Y]").astype("datetime64[D]")
        doy = int((t - yr_start).astype(int)) + 1          # 1-based
        angle = 2.0 * np.pi * doy / 365.0
        return np.array([np.sin(angle), np.cos(angle)], dtype="float32")


# ---------------------------------------------------------------------------
# Transform  (single source of truth for normalization)
# ---------------------------------------------------------------------------

class Transform:
    """
    Normalize / denormalize physical arrays using supplied statistics.

    Convention:  z = (x - mean) / std   (per channel, broadcast over grid)
    """

    def __init__(self, stats_path: Path = STATS_PATH) -> None:
        with open(stats_path) as f:
            s = json.load(f)
        self.mean = np.array(s["mean"], dtype="float32")    # [6]
        self.std  = np.array(s["std"],  dtype="float32")    # [6]

    # shapes: [..., 6, H, W]
    def normalize(self, x: np.ndarray) -> np.ndarray:
        m = self.mean[:, None, None]
        s = self.std[:, None, None]
        return (x - m) / s

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        m = self.mean[:, None, None]
        s = self.std[:, None, None]
        return z * s + m

    # --- single-channel helpers (index c in 0..5) ---
    def normalize_channel(self, x: np.ndarray, c: int) -> np.ndarray:
        return (x - self.mean[c]) / self.std[c]

    def denormalize_channel(self, z: np.ndarray, c: int) -> np.ndarray:
        return z * self.std[c] + self.mean[c]


# ---------------------------------------------------------------------------
# WindowIndex
# ---------------------------------------------------------------------------

class WindowIndex:
    """
    Build non-overlapping, chronologically clean windows.

    Each window:  3 consecutive input days + up to `max_horizon` target days.
    All day indices belong to the same partition (no cross-split leakage).

    Parameters
    ----------
    indices : sorted array of day-indices that belong to this split.
    n_inputs : number of input (history) days (always 3 per PRD §4).
    max_horizon : maximum forecast horizon (up to 14; default 10 for training).
    """

    def __init__(
        self,
        indices: np.ndarray,
        n_inputs: int = 3,
        max_horizon: int = 10,
    ) -> None:
        self.n_inputs = n_inputs
        self.max_horizon = max_horizon
        total_len = n_inputs + max_horizon

        windows: List[Tuple[np.ndarray, np.ndarray]] = []
        i = 0
        while i + total_len <= len(indices):
            block = indices[i : i + total_len]
            # verify strictly consecutive (no gaps across the split boundary)
            if np.all(np.diff(block) == 1):
                inp_idx = block[:n_inputs]
                tgt_idx = block[n_inputs:]
                windows.append((inp_idx, tgt_idx))
            i += 1  # sliding window (stride 1)

        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (input_indices [3], target_indices [max_horizon])."""
        return self.windows[i]
