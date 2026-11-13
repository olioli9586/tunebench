"""int8 quantization variants, compared on validation (never test).

    python scripts/quant_variants.py artifacts/onnx/modernbert-seed2

Quantizes the exported float32 model two ways, per-tensor scales (ONNX Runtime's
default) and per-channel scales (what tunebench.export ships, chosen with this
script), and scores each against float32 on the validation split. Writes results/serving/quant-variants-<name>.json.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

from tunebench.data import ROOT, load
from tunebench.export import evaluate
from tunebench.metrics import scores


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
        variants = {"per_tensor": {}, "per_channel": {"per_channel": True}}
        for name, options in variants.items():
            out = work / f"{name}.onnx"
            quantize_dynamic(str(work / "prepared.onnx"), str(out), weight_type=QuantType.QInt8, **options)
            predicted, _ = evaluate(out, val.texts)
            report[name] = {"size_mb": round(out.stat().st_size / 1e6, 1), **scores(val.labels, predicted)}
    predicted, _ = evaluate(a.model_dir / "model.onnx", val.texts)
    report["fp32"] = scores(val.labels, predicted)
    print(json.dumps(report, indent=1))
    out = ROOT / "results" / "serving" / f"quant-variants-{a.model_dir.name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
