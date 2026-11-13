"""int8 quantization variants, compared on validation (never test).

    python scripts/quant_variants.py artifacts/onnx/modernbert-seed2

Quantizes the exported float32 model several ways and scores each against
float32 on the validation split, with the single-message latency on this CPU:
- per-tensor or per-channel scales (one per matrix, or one per output row);
- signed int8 weights (ONNX Runtime's default, "U8S8" with unsigned activations),
  the same with 7-bit weights (reduce_range), or unsigned weights ("U8U8").
On x86 CPUs without VNNI, ONNX Runtime computes U8S8 with an instruction that
adds pairs of products into 16 bits and clamps what doesn't fit; 7-bit or
unsigned weights can't overflow it. ARM CPUs (this Apple one) never clamp, so
the scores here carry over to any x86 server only for the variants that can't
overflow. Full 8-bit per-channel weights scored 93.6% on validation here and
77% on the test set on the Vercel deployment (x86).
Writes results/serving/quant-variants-<name>.json.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

from tunebench.data import ROOT, load
from tunebench.export import evaluate
from tunebench.metrics import percentiles, scores
from tunebench.serve import Classifier

VARIANTS = {
    "per_tensor": {"weight_type": QuantType.QInt8},
    "per_channel": {"weight_type": QuantType.QInt8, "per_channel": True},
    "per_channel_reduce_range": {"weight_type": QuantType.QInt8, "per_channel": True, "reduce_range": True},
    "per_tensor_u8u8": {"weight_type": QuantType.QUInt8},
    "per_channel_u8u8": {"weight_type": QuantType.QUInt8, "per_channel": True},
}
CLAMPS_ON_X86 = {"per_tensor", "per_channel"}  # without VNNI


def p50_ms(model_file: Path, texts: list[str]) -> float:
    """Median single-message latency with one thread."""
    import time

    classifier = Classifier(model_file.parent, model_file.name, threads=1)
    classifier.logits(texts[:20])
    times = []
    for text in texts:
        started = time.perf_counter()
        classifier.logits([text])
        times.append(1000 * (time.perf_counter() - started))
    return round(percentiles(times)["p50"], 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    a = parser.parse_args()
    val = load().val
    report: dict[str, object] = {"model": a.model_dir.name, "split": "validation", "n": len(val)}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("tokenizer.json", "labels.json"):
            shutil.copy(a.model_dir / name, work / name)
        quant_pre_process(str(a.model_dir / "model.onnx"), str(work / "prepared.onnx"))
        for name, options in VARIANTS.items():
            out = work / f"{name}.onnx"
            quantize_dynamic(str(work / "prepared.onnx"), str(out), **options)
            predicted, _ = evaluate(out, val.texts)
            report[name] = {
                "size_mb": round(out.stat().st_size / 1e6, 1),
                "clamps_on_x86_without_vnni": name in CLAMPS_ON_X86,
                **scores(val.labels, predicted),
                "p50_ms_1_thread": p50_ms(out, val.texts[:300]),
            }
            print(name, report[name])
    predicted, _ = evaluate(a.model_dir / "model.onnx", val.texts)
    fp32_ms = p50_ms(a.model_dir / "model.onnx", val.texts[:300])
    report["fp32"] = {**scores(val.labels, predicted), "p50_ms_1_thread": fp32_ms}
    print(json.dumps(report, indent=1))
    out = ROOT / "results" / "serving" / f"quant-variants-{a.model_dir.name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
