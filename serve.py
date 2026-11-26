"""Offline judge entry point: ``python serve.py --port 8000``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from deadlock_monsoon.contract import ContractError
from deadlock_monsoon.engine import PersistenceEngine
from deadlock_monsoon.service import run_request


def load_engine():
    bundle = Path(os.environ.get("DEADLOCK_MODEL_BUNDLE", ROOT / "artifacts" / "primary"))
    try:
        from deadlock_monsoon.learned_engine import GRASTEngine

        return GRASTEngine(bundle)
    except Exception as exc:
        print(f"learned model unavailable; using persistence fallback: {exc}", file=sys.stderr)
        return PersistenceEngine()


ENGINE = load_engine()
app = FastAPI(title="Deadlock Monsoon", docs_url=None, redoc_url=None)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": ENGINE.name}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    started = time.perf_counter()
    try:
        body = await request.json()
        rollout = run_request(ENGINE, body)
    except ContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "rollout": rollout.tolist(),
        "inference_ms": (time.perf_counter() - started) * 1000.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

