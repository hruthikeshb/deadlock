"""Reproducible HTTP cold-start, latency, memory, size, and soak benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]


def post(url: str, body: dict, timeout: float = 10.0) -> tuple[dict, float, int]:
    encoded = json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return json.loads(payload), elapsed_ms, len(payload)


def valid_payload(steps: int, shocked: bool = False) -> dict:
    history = np.zeros((3, 6, 32, 36), dtype=np.float32)
    history[:, 0] = 298.0
    history[:, 1] = 2.0
    history[:, 4] = 100_000.0
    history[:, 5] = 30.0
    perturbation = None
    if shocked:
        perturbation = np.zeros((6, 32, 36), dtype=np.float32)
        perturbation[0, 14:17, 16:19] = 1.0
    return {
        "init_states": history.tolist(),
        "doy_sincos": [0.0, 1.0],
        "steps": steps,
        "perturbation": None if perturbation is None else perturbation.tolist(),
    }


def rss_bytes(process: psutil.Process) -> int:
    processes = [process] + process.children(recursive=True)
    total = 0
    for item in processes:
        try:
            total += item.memory_info().rss
        except psutil.Error:
            pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"
    command = [sys.executable, "serve.py", "--port", str(args.port)]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.perf_counter()
    server = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    process = psutil.Process(server.pid)
    try:
        health = None
        for _ in range(120):
            if server.poll() is not None:
                error = server.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"server exited during startup: {error}")
            try:
                health, _, _ = post(f"{base_url}/health", {})
                break
            except Exception:
                time.sleep(0.25)
        if health is None:
            raise RuntimeError("server did not become ready")
        cold_start_ms = (time.perf_counter() - started) * 1000.0
        peak_rss = rss_bytes(process)
        horizon_results = {}
        for steps in (1, 10, 14):
            response, elapsed_ms, response_bytes = post(
                f"{base_url}/simulate", valid_payload(steps)
            )
            rollout = np.asarray(response["rollout"], dtype=np.float32)
            peak_rss = max(peak_rss, rss_bytes(process))
            horizon_results[str(steps)] = {
                "http_ms": elapsed_ms,
                "reported_inference_ms": float(response["inference_ms"]),
                "response_bytes": response_bytes,
                "shape": list(rollout.shape),
                "finite": bool(np.isfinite(rollout).all()),
                "sha256": hashlib.sha256(rollout.tobytes()).hexdigest(),
            }
        all_horizons = {}
        for steps in range(1, 15):
            if str(steps) in horizon_results:
                item = horizon_results[str(steps)]
                all_horizons[str(steps)] = item["shape"] == [steps, 6, 32, 36] and item["finite"]
                continue
            response, _, _ = post(f"{base_url}/simulate", valid_payload(steps))
            rollout = np.asarray(response["rollout"], dtype=np.float32)
            peak_rss = max(peak_rss, rss_bytes(process))
            all_horizons[str(steps)] = bool(
                rollout.shape == (steps, 6, 32, 36) and np.isfinite(rollout).all()
            )
        soak = []
        for index in range(8):
            response, elapsed_ms, response_bytes = post(
                f"{base_url}/simulate", valid_payload(10, shocked=bool(index % 2))
            )
            rollout = np.asarray(response["rollout"], dtype=np.float32)
            peak_rss = max(peak_rss, rss_bytes(process))
            soak.append({
                "index": index,
                "shocked": bool(index % 2),
                "http_ms": elapsed_ms,
                "response_bytes": response_bytes,
                "finite": bool(np.isfinite(rollout).all()),
                "sha256": hashlib.sha256(rollout.tobytes()).hexdigest(),
            })
        ordinary_hashes = {item["sha256"] for item in soak if not item["shocked"]}
        shocked_hashes = {item["sha256"] for item in soak if item["shocked"]}
        report = {
            "health": health,
            "cold_start_ms": cold_start_ms,
            "peak_rss_bytes": peak_rss,
            "horizons": horizon_results,
            "all_horizons_1_through_14": all_horizons,
            "soak": soak,
            "acceptance": {
                "ready_under_120_seconds": cold_start_ms < 120_000,
                "all_requests_under_5_seconds": all(
                    item["http_ms"] < 5_000 for item in horizon_results.values()
                ) and all(item["http_ms"] < 5_000 for item in soak),
                "rss_under_6_gb": peak_rss < 6 * 1024**3,
                "responses_under_80_mb": all(
                    item["response_bytes"] < 80 * 1024**2 for item in horizon_results.values()
                ) and all(item["response_bytes"] < 80 * 1024**2 for item in soak),
                "all_finite": all(item["finite"] for item in horizon_results.values())
                and all(item["finite"] for item in soak),
                "all_horizons_1_through_14": all(all_horizons.values()),
                "deterministic_interleaved": len(ordinary_hashes) == 1
                and len(shocked_hashes) == 1
                and ordinary_hashes != shocked_hashes,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "cold_start_ms": cold_start_ms,
            "peak_rss_bytes": peak_rss,
            "horizon_http_ms": {
                key: value["http_ms"] for key, value in horizon_results.items()
            },
            "acceptance": report["acceptance"],
        }, indent=2, sort_keys=True))
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
