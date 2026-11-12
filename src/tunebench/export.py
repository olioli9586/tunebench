"""Export a fine-tuned classifier to ONNX, then quantize its weights to int8.

    python -m tunebench.export artifacts/modernbert-seed1

writes artifacts/onnx/<name>/: model.onnx (float32), model.int8.onnx (dynamic
int8 quantization: weights stored as 8-bit integers, activations quantized on
the fly), tokenizer.json and labels.json, everything the server needs. It then
scores both files on the test set with ONNX Runtime on the CPU and writes
sizes and accuracy to results/serving/onnx-<name>.json, so a drop in accuracy
from quantization is measured rather than assumed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import ROOT
from .metrics import scores

SERVING_RESULTS = ROOT / "results" / "serving"
INPUTS = ("input_ids", "attention_mask", "token_type_ids")


def export(model_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    import torch
    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    sample = tokenizer(["how do I top up my card?", "my card has not arrived yet"], padding=True, return_tensors="pt")
    names = [n for n in INPUTS if n in sample]
    fp32 = out_dir / "model.onnx"
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence")
    torch.onnx.export(
        model,
        kwargs={n: sample[n] for n in names},
        f=str(fp32),
        input_names=names,
        output_names=["logits"],
        dynamic_shapes={n: {0: batch, 1: seq} for n in names},
        dynamo=True,
        external_data=False,
    )
    prepared = out_dir / "model.prepared.onnx"
    quant_pre_process(str(fp32), str(prepared))
    int8 = out_dir / "model.int8.onnx"
    quantize_dynamic(str(prepared), str(int8), weight_type=QuantType.QInt8)
    prepared.unlink()

    backend = tokenizer.backend_tokenizer
    backend.enable_padding(pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token)
    backend.save(str(out_dir / "tokenizer.json"))
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1) + "\n")
    return fp32, int8


def evaluate(model_file: Path, texts: list[str], batch_size: int = 64) -> tuple[list[int], float]:
    """Predicted intents for texts, and the seconds it took, with ONNX Runtime on the CPU."""
    from .serve import Classifier

    classifier = Classifier(model_file.parent, model_file.name)
    started = time.perf_counter()
    predicted: list[int] = []
    for i in range(0, len(texts), batch_size):
        predicted.extend(np.argmax(classifier.logits(texts[i : i + batch_size]), axis=-1).tolist())
    return predicted, time.perf_counter() - started


def main() -> None:
    from .data import load

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    a = parser.parse_args()
    name = a.model_dir.name
    out_dir = ROOT / "artifacts" / "onnx" / name
    fp32, int8 = export(a.model_dir, out_dir)
    data = load()
    report: dict[str, object] = {"model": name}
    for label, path in (("fp32", fp32), ("int8", int8)):
        predicted, seconds = evaluate(path, data.test.texts)
        report[label] = {
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            **scores(data.test.labels, predicted),
            "test_seconds_batch64": round(seconds, 1),
            "predictions": predicted,
        }
        print(f"{label}: {report[label]['size_mb']} MB, accuracy {report[label]['accuracy']:.4f}")
    SERVING_RESULTS.mkdir(parents=True, exist_ok=True)
    (SERVING_RESULTS / f"onnx-{name}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
