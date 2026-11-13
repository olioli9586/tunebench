"""Publishes the fine-tuned model to the Hugging Face Hub.

    .venv/bin/hf auth login                      # once, with a token that can write
    python scripts/publish.py modernbert-seed2   # the run exported with tunebench.export

Creates or updates the public repo <user>/modernbert-base-banking77: the model
in transformers format (safetensors), onnx/ with the float32 and int8 files,
and a model card whose numbers come from results/. The live demo runs on
Vercel instead of a Hugging Face Space (Docker Spaces need a paid plan); its
image downloads the int8 model from this repo (Dockerfile.vercel).
"""

import argparse
import json

from huggingface_hub import HfApi

from tunebench.data import ROOT
from tunebench.train import RECIPES

MODEL_NAME = "modernbert-base-banking77"
GITHUB = "https://github.com/olioli9586/tunebench"
DEMO = "https://tunebench-henna.vercel.app"


def model_card(run: dict, onnx: dict, latency: dict | None, model_repo: str) -> str:
    recipe = RECIPES["modernbert"]
    int8, fp32 = onnx["int8"], onnx["fp32"]
    speed = ""
    if latency:
        http = latency["int8"]["http"]
        speed = (
            f"\nServed as int8 ONNX behind FastAPI with 2 CPU threads ({latency['machine']}): p50 "
            f"{http['p50']:.1f} ms, p95 {http['p95']:.1f} ms per request.\n"
        )
    return f"""---
license: apache-2.0
base_model: {recipe.checkpoint}
datasets:
- PolyAI/banking77
language:
- en
pipeline_tag: text-classification
tags:
- intent-classification
- banking
- onnx
---

# ModernBERT-base fine-tuned on Banking77

Classifies a message to a bank's customer support into one of 77 intents
(`card_arrival`, `lost_or_stolen_card`, `top_up_failed`, ...). Part of
[TuneBench]({GITHUB}), which compares it with a TF-IDF baseline, a LoRA-tuned
Qwen3-0.6B and Claude Haiku 5.5 on the same test set.

**Try it:** [{DEMO}]({DEMO}) (the int8 file below behind FastAPI on a
Vercel CPU; the first request after a few idle minutes takes a few seconds).

## Results on the Banking77 test split (3,080 messages, 40 per intent)

| File | Size | Accuracy | Macro-F1 |
|---|---|---|---|
| PyTorch (safetensors in this repo) | {fp32["size_mb"]:.0f} MB | {100 * run["accuracy"]:.1f}% | {run["macro_f1"]:.4f} |
| `onnx/model.onnx` (float32) | {fp32["size_mb"]:.0f} MB | {100 * fp32["accuracy"]:.1f}% | {fp32["macro_f1"]:.4f} |
| `onnx/model.int8.onnx` (dynamic int8, per-channel, 7-bit weights) | {int8["size_mb"]:.0f} MB | {100 * int8["accuracy"]:.1f}% | {int8["macro_f1"]:.4f} |
{speed}
int8 accuracy depends on the CPU. The numbers above are on an Apple M5 Pro,
where ONNX Runtime runs the int8 matrix multiplies in an ARM-specific fused
kernel. On x86 the int8 file scores about 91% (91.2% on an Intel Xeon, 91.3%
on the live demo); float32 gives the same answers everywhere. The weights use 7
of their 8 bits on purpose: with full 8-bit weights, ONNX Runtime on x86 CPUs
without VNNI overflows 16-bit sums, and the first deployment scored 77%.

This is seed {run["seed"]} of three; the repo reports the mean and spread over all three.

## Training

Full fine-tune of `{recipe.checkpoint}` (revision `{recipe.revision[:7]}`) with a
classification head: learning rate {recipe.learning_rate}, batch {recipe.batch_size}, up to
{recipe.epochs} epochs with early stopping on validation macro-F1 (best epoch {run["best_epoch"]:.0f}),
10% linear warmup, weight decay 0.01, bfloat16 mixed precision on an Apple M5 Pro GPU.
Validation is a stratified 10% of the Banking77 train split; the test split was
used once, after training. Code: [{GITHUB}]({GITHUB}).

## Use

```python
from transformers import pipeline

classify = pipeline("text-classification", model="{model_repo}")
classify("I still haven't received my new card")
```

The `onnx/` folder holds `tokenizer.json` and `labels.json` for serving
without PyTorch; see `src/tunebench/serve.py` in the code repo.

## Limits

Banking77 is single-turn English messages about a narrow set of retail-banking
topics. The model always answers one of the 77 intents, even for messages that
fit none of them; it has no "other" class. The most common confusions are
between near-duplicate intents (see the error analysis in the code repo).

Dataset: Banking77 (Casanueva et al., 2020), CC BY 4.0.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="e.g. modernbert-seed2 (artifacts/<run> and artifacts/onnx/<run> must exist)")
    a = parser.parse_args()
    run = json.loads((ROOT / "results" / f"{a.run}.json").read_text())
    onnx = json.loads((ROOT / "results" / "serving" / f"onnx-{a.run}.json").read_text())
    latency_file = ROOT / "results" / "serving" / f"latency-{a.run}.json"
    latency = json.loads(latency_file.read_text()) if latency_file.exists() else None

    api = HfApi()
    user = api.whoami()["name"]
    model_repo = f"{user}/{MODEL_NAME}"

    api.create_repo(model_repo, exist_ok=True)
    api.upload_folder(repo_id=model_repo, folder_path=ROOT / "artifacts" / a.run, commit_message=f"Model ({a.run})")
    api.upload_folder(
        repo_id=model_repo,
        folder_path=ROOT / "artifacts" / "onnx" / a.run,
        path_in_repo="onnx",
        commit_message="ONNX float32 and int8",
    )
    api.upload_file(
        repo_id=model_repo,
        path_or_fileobj=model_card(run, onnx, latency, model_repo).encode(),
        path_in_repo="README.md",
        commit_message="Model card",
    )

    print(f"Model: https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
