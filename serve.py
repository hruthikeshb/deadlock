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

from src.data.dataset import Transform, GRID_H, GRID_W
from src.features.geo import build_geo_features
from src.models.grast import GRAST, FallbackCNN, C_IN
from src.rollout.engine import RolloutEngine

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

# Bundle only the static geography needed for inference, not the historical archive.
_static = np.load(ROOT / "artifacts" / "serve_static.npy", allow_pickle=False).astype("float32")
_lats = np.load(ROOT / "artifacts" / "serve_lats.npy", allow_pickle=False).astype("float32")
_lons = np.load(ROOT / "artifacts" / "serve_lons.npy", allow_pickle=False).astype("float32")
if _static.shape != (2, GRID_H, GRID_W):
    raise ValueError(f"Expected static fields with shape {(2, GRID_H, GRID_W)}, got {_static.shape}")
if _lats.shape != (GRID_H,) or _lons.shape != (GRID_W,):
    raise ValueError(f"Expected latitude/longitude shapes {(GRID_H,)} and {(GRID_W,)}")

_geo_np = build_geo_features(_static[0], _static[1], _lats, _lons)
_transform = Transform(ROOT / "data" / "stats.json")
_engine = RolloutEngine(_model, _transform, _geo_np, torch.device("cpu"))

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
