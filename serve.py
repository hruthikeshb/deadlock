"""
serve.py — Production inference server for "Multiverse of Monsoon" (Track 2).
=============================================================================

Loads the trained GRAST model from artifacts/primary_grast.pt at startup and
serves autoregressive 10-step rollouts via the R1 API contract.

Run:  python serve.py --port 8000
Validate:  python -m monsoon_r1.schema_validator http://localhost:8000

Rules enforced:
  - listen on 0.0.0.0
  - weights loaded once at import time (never downloaded at runtime)
  - returns physical units (K, mm/day, m/s, Pa, kg/m²)
  - tp and tcwv clamped >= 0 at every rollout step
  - 10-step rollout < 5 s on 4 CPU cores / 6 GB RAM
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import Transform, N_CHANNELS, GRID_H, GRID_W, CHANNELS
from src.features.geo import build_geo_features
from src.models.grast import GRAST, FallbackCNN, C_IN
from src.rollout.engine import RolloutEngine
from src.data.hdf5_reader import load_nc_arrays

# ---------------------------------------------------------------------------
# Load model ONCE at import time
# ---------------------------------------------------------------------------

_CKPT_PATH = ROOT / "artifacts" / "primary_grast.pt"

ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)

_model_name: str = ckpt.get("model_name", "primary_grast")
_base_ch: int    = ckpt.get("base_ch", 64)
_c_in: int       = ckpt.get("c_in", C_IN)
_model_class: str = ckpt.get("model_class", "primary")

if _model_class == "primary":
    _model = GRAST(c_in=_c_in, base_ch=_base_ch)
else:
    _model = FallbackCNN(c_in=_c_in, base_ch=_base_ch)

_model.load_state_dict(ckpt["state_dict"])
_model.eval()

# Load static fields for geo features (land-sea mask + orography)
_nc_arrays  = load_nc_arrays(ROOT / "data" / "india_climate_train.nc")
_static     = _nc_arrays["static"].astype("float32")          # [2, 32, 36]
_lats       = _nc_arrays["lat"].astype("float32")              # [32]
_lons       = _nc_arrays["lon"].astype("float32")              # [36]
_geo_np     = build_geo_features(_static[0], _static[1], _lats, _lons)  # [6,32,36]

_transform  = Transform()
_engine     = RolloutEngine(_model, _transform, _geo_np, torch.device("cpu"))

# Fixed historical date from the locked test split (2020-07-15, peak monsoon)
_FIXED_DATE_STR = "2020-07-15"
import xarray as _xr
_ds_meta = _xr.open_dataset(ROOT / "data" / "india_climate_train.nc", engine="h5netcdf")
_dates = _ds_meta["time"].values.astype("datetime64[D]")
_ds_meta.close()
_date_matches = np.where(_dates == np.datetime64(_FIXED_DATE_STR, "D"))[0]
_fixed_idx = int(_date_matches[0]) if len(_date_matches) > 0 else int(len(_dates) - 11)
_fixed_date = str(_dates[_fixed_idx])
_fixed_init_states = _nc_arrays["data"][_fixed_idx - 2 : _fixed_idx + 1].astype("float32")  # [3, 6, 32, 36]
_yr_start = _dates[_fixed_idx].astype("datetime64[Y]").astype("datetime64[D]")
_doy = int((_dates[_fixed_idx] - _yr_start).astype(int)) + 1
_angle = 2.0 * np.pi * _doy / 365.0
_fixed_doy_sincos = [float(np.sin(_angle)), float(np.cos(_angle))]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _arr(x) -> np.ndarray:
    """Accept nested lists (contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        import base64
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


@app.post("/health")
def health():
    return {"status": "ready", "model_name": _model_name}


@app.get("/initial-conditions")
def initial_conditions():
    return {
        "date": _fixed_date,
        "init_states": _fixed_init_states.tolist(),
        "doy_sincos": _fixed_doy_sincos,
        "metadata": {
            "lats": _lats.tolist(),
            "lons": _lons.tolist(),
            "channels": CHANNELS,
            "units": {
                "t2m": "K",
                "tp": "mm/day",
                "u10": "m/s",
                "v10": "m/s",
                "msl": "Pa",
                "tcwv": "kg/m2",
            },
            "description": f"ERA5 / IMD 3-day observational state ending {_fixed_date} (32x36 grid)",
        },
    }


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0   = time.perf_counter()
    body = await request.json()

    init_states: np.ndarray = _arr(body["init_states"])           # [3, 6, 32, 36]
    steps: int               = int(body["steps"])
    doy_sincos               = body["doy_sincos"]                 # [sin, cos]
    pert_raw                 = body.get("perturbation")
    perturbation: np.ndarray | None = None if pert_raw is None else _arr(pert_raw)  # [6,32,36]

    doy_arr = np.asarray(doy_sincos, dtype="float32")

    rollout: np.ndarray = _engine.rollout(
        init_states  = init_states,
        doy_sincos   = doy_arr,
        steps        = steps,
        perturbation = perturbation,
    )  # [steps, 6, 32, 36] in physical units

    # Extra physical safeguards beyond what RolloutEngine already enforces
    rollout[:, 1] = np.maximum(rollout[:, 1], 0.0)   # tp   >= 0
    rollout[:, 5] = np.maximum(rollout[:, 5], 0.0)   # tcwv >= 0

    inference_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "rollout":      np.asarray(rollout, dtype="f4").tolist(),
        "inference_ms": inference_ms,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
