"""CPU latency of a fine-tuned model in PyTorch, before any export.

    python scripts/torch_latency.py artifacts/qwen3-lora-r16-seed0

One message at a time on 2 CPU threads (the same budget as the HTTP numbers in
scripts/latency.py), 200 test messages after 10 warm-up calls. This is the
common yardstick for every fine-tuned model, including the LoRA model, which is
not exported to ONNX. Writes results/serving/torch-latency-<name>.json.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tunebench.data import ROOT, load
from tunebench.metrics import percentiles

THREADS = 2
MESSAGES = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    a = parser.parse_args()
    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_dir, dtype=torch.float32).eval()
    texts = [load().test.texts[i] for i in np.random.default_rng(0).permutation(3080)[: MESSAGES + 10]]
    times = []
    with torch.inference_mode():
        for i, text in enumerate(texts):
            started = time.perf_counter()
            model(**tokenizer([text], return_tensors="pt"))
            if i >= 10:
                times.append((time.perf_counter() - started) * 1000)
    report = {
        "model": a.model_dir.name,
        "machine": f"{platform.system()} {platform.machine()}",
        "threads": THREADS,
        "params": sum(p.numel() for p in model.parameters()),
        "latency_ms": percentiles(times),
    }
    print(json.dumps(report, indent=1))
    out = ROOT / "results" / "serving" / f"torch-latency-{a.model_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
