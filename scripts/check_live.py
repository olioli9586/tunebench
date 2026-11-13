"""Scores the deployed demo on the test set over HTTP, against the local int8 file's answers.

    python scripts/check_live.py https://tunebench-henna.vercel.app

Sends each of the 3,080 test messages to POST /predict (4 at a time) and writes
results/serving/live-demo.json: accuracy, how many answers match the local
int8 file's (results/serving/onnx-<model>.json), and the predictions.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from tunebench.data import ROOT, load

SERVING = ROOT / "results" / "serving"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--model", default="modernbert-seed2")
    a = parser.parse_args()
    data = load()
    local = json.loads((SERVING / f"onnx-{a.model}.json").read_text())["int8"]["predictions"]
    client = httpx.Client(timeout=60)
    health = client.get(f"{a.url}/health").json()

    def ask(text: str) -> int:
        for attempt in range(3):
            try:
                response = client.post(f"{a.url}/predict", json={"text": text})
                response.raise_for_status()
                return data.names.index(response.json()["intent"])
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                time.sleep(2)
        raise AssertionError

    started = time.time()
    with ThreadPoolExecutor(4) as pool:
        predicted = list(pool.map(ask, data.test.texts))
    accuracy = sum(p == t for p, t in zip(predicted, data.test.labels, strict=True)) / len(predicted)
    same = sum(p == q for p, q in zip(predicted, local, strict=True))
    print(f"{len(predicted)} messages in {time.time() - started:.0f} s: accuracy {accuracy:.4f}, same as local {same}")
    report = {"url": a.url, "health": health, "accuracy": accuracy, "same_as_local_int8": same}
    report["predictions"] = predicted
    (SERVING / "live-demo.json").write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
