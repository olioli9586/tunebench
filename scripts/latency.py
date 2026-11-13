"""CPU latency and throughput of the exported model, float32 vs int8.

    python scripts/latency.py artifacts/onnx/modernbert-seed2

Two views, both on test-set messages:
- In process (ONNX Runtime alone): one message at a time, with 1 thread,
  2 threads (a small cloud box or a free Hugging Face Space) and every core.
- Over HTTP (the FastAPI server under uvicorn): 500 sequential requests for
  p50/p95 with THREADS=2; then throughput on about two cores' worth of work,
  THREADS=1 with 2 clients sending at once (each request runs on one thread),
  which is what the self-hosting cost estimate in RESULTS.md divides by.

Writes results/serving/latency-<name>.json.
"""

import argparse
import asyncio
import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from tunebench.data import ROOT, load
from tunebench.metrics import percentiles
from tunebench.serve import Classifier

WARMUP = 20
SINGLE = 500
CONCURRENT_REQUESTS = 1000


def in_process(model_dir: Path, file: str, texts: list[str], threads: int | None) -> dict:
    classifier = Classifier(model_dir, file, threads)
    for text in texts[:WARMUP]:
        classifier.logits([text])
    times = []
    for text in texts[:SINGLE]:
        started = time.perf_counter()
        classifier.logits([text])
        times.append((time.perf_counter() - started) * 1000)
    return percentiles(times)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def concurrent(url: str, texts: list[str], clients: int) -> float:
    queue = list(texts)

    async def client(http: httpx.AsyncClient) -> None:
        while queue:
            (await http.post(url, json={"text": queue.pop()})).raise_for_status()

    async with httpx.AsyncClient(timeout=30) as http:
        started = time.perf_counter()
        await asyncio.gather(*(client(http) for _ in range(clients)))
        return len(texts) / (time.perf_counter() - started)


@contextlib.contextmanager
def server(model_dir: Path, file: str, threads: int):
    """The FastAPI server in a subprocess, ready and warmed up; yields its base URL."""
    port = free_port()
    env = os.environ | {"MODEL_DIR": str(model_dir), "MODEL_FILE": file, "THREADS": str(threads)}
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tunebench.serve:app", "--port", str(port), "--log-level", "warning"],
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{base}/health").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.2)
        with httpx.Client() as http:
            for text in [f"warm up {i}" for i in range(WARMUP)]:
                http.post(f"{base}/predict", json={"text": text})
        yield base
    finally:
        process.terminate()
        process.wait()


def over_http(model_dir: Path, file: str, texts: list[str]) -> dict:
    with server(model_dir, file, threads=2) as base, httpx.Client() as http:
        times = []
        for text in texts[:SINGLE]:
            started = time.perf_counter()
            http.post(f"{base}/predict", json={"text": text}).raise_for_status()
            times.append((time.perf_counter() - started) * 1000)
    with server(model_dir, file, threads=1) as base:
        rps = asyncio.run(concurrent(f"{base}/predict", texts[:CONCURRENT_REQUESTS], clients=2))
    return {**percentiles(times), "requests_per_second_2_cores": round(rps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    a = parser.parse_args()
    # A fixed shuffle so short and long messages are mixed (the test split is sorted by intent).
    texts = [load().test.texts[i] for i in np.random.default_rng(0).permutation(3080)]
    report: dict[str, object] = {
        "model": a.model_dir.name,
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores",
        "date": time.strftime("%Y-%m-%d"),
    }
    for label, file in (("fp32", "model.onnx"), ("int8", "model.int8.onnx")):
        report[label] = {
            "size_mb": round((a.model_dir / file).stat().st_size / 1e6, 1),
            "in_process": {f"threads_{t or 'all'}": in_process(a.model_dir, file, texts, t) for t in (1, 2, None)},
            "http": over_http(a.model_dir, file, texts),
        }
        print(label, json.dumps(report[label], indent=1))
    out = ROOT / "results" / "serving" / f"latency-{a.model_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
