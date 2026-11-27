"""Sanity-check data/india_climate_train.nc and data/stats.json.
Run from the repo root:  python tests/test_dataset.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
NC_PATH = ROOT / "data" / "india_climate_train.nc"
STATS_PATH = ROOT / "data" / "stats.json"
CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]

failures = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def main():
    check(NC_PATH.exists(), f"found {NC_PATH}")
    check(STATS_PATH.exists(), f"found {STATS_PATH}")
    if failures:
        return 1

    ds = xr.open_dataset(NC_PATH, engine="h5netcdf")
    print("\n=== Dataset summary ===")
    print(ds)

    # ---- main data variable ----
    check("data" in ds, "variable 'data' exists")
    if "data" not in ds:
        return 1
    data = ds["data"]
    check(tuple(data.dims) == ("time", "channel", "lat", "lon"),
          f"dims are (time, channel, lat, lon): {tuple(data.dims)}")
    x = data.values.astype("float64")
    check(x.shape[1:] == (6, 32, 36), f"grid shape is (6, 32, 36): {x.shape[1:]}")
    print(f"      total days: {x.shape[0]}")

    check(list(ds["channel"].values) == CHANNELS,
          f"channel order is {CHANNELS}: {list(ds['channel'].values)}")
    check(np.allclose(ds["lat"].values, np.arange(6.5, 37.5 + 1e-9, 1.0)),
          "lat centres are 6.5 .. 37.5")
    check(np.allclose(ds["lon"].values, np.arange(65.5, 100.5 + 1e-9, 1.0)),
          "lon centres are 65.5 .. 100.5")

    t = ds["time"].values
    print(f"      date range: {str(t[0])[:10]} -> {str(t[-1])[:10]}")
    check(bool((np.diff(t) == np.timedelta64(1, "D")).all()),
          "time axis is strictly daily with no gaps")

    # ---- values ----
    check(bool(np.isfinite(x).all()), "no NaN / Inf in data")
    check(x[:, 1].min() >= 0, f"rain (tp) never negative (min {x[:, 1].min():.6f})")
    check(230 < x[:, 0].min() and x[:, 0].max() < 330,
          f"t2m in a plausible range in K ({x[:, 0].min():.1f} .. {x[:, 0].max():.1f})")
    check(90000 < x[:, 4].min() and x[:, 4].max() < 110000,
          f"msl in a plausible range in Pa ({x[:, 4].min():.0f} .. {x[:, 4].max():.0f})")
    check(x[:, 5].min() >= 0, f"tcwv non-negative (min {x[:, 5].min():.2f})")

    print("\n=== Per-channel statistics (all days, all cells) ===")
    print(f"{'channel':8s} {'min':>12s} {'max':>12s} {'mean':>12s} {'std':>12s}")
    for i, name in enumerate(CHANNELS):
        c = x[:, i]
        print(f"{name:8s} {c.min():12.3f} {c.max():12.3f} {c.mean():12.3f} {c.std():12.3f}")

    print("\nFirst day, channel means:",
          {n: round(float(x[0, i].mean()), 3) for i, n in enumerate(CHANNELS)})

    # ---- static fields ----
    print()
    check("static" in ds, "variable 'static' exists")
    if "static" in ds:
        s = ds["static"].values
        check(s.shape == (2, 32, 36), f"static shape is (2, 32, 36): {s.shape}")

