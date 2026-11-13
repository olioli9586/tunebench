"""The published ONNX files on this machine's CPU: does int8 give the answers it gives elsewhere?

    python scripts/cpu_check.py                      # writes results/serving/cpu-<machine>.json

Downloads the published files from the Hugging Face model repo at pinned
commits (float32, the first 8-bit int8 file, the shipped 7-bit one), builds
an int8 variant with unsigned weights from the float32 file, and scores each
on the test set with ONNX Runtime on the CPU. The shipped file is scored a
second time with ONNX Runtime's graph fusions turned off, because some CPUs
run the fused int8 operator with their own kernel. Prints the CPU model and
its vector extensions (AVX2, AVX-512, VNNI). Run on x86 Linux in a Vercel
Sandbox for the x86 numbers in RESULTS.md.
"""

import json
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

from tunebench.data import ROOT, load
from tunebench.serve import Classifier

REPO = "Oliilo123/modernbert-base-banking77"
SHIPPED = "3bf79a0d8b0df01a943b142ad4cb189a0cca12ad"  # 7-bit per-channel int8
FIRST_INT8 = "4910c6344ea6567dec710c7726c8250e0ebad054"  # 8-bit per-channel int8
FEATURES = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8", "asimddp", "i8mm", "sme2")


def cpu() -> tuple[str, list[str]]:
    if platform.system() == "Darwin":
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        flags = subprocess.run(["sysctl", "-a"], capture_output=True, text=True).stdout
        have = [f for f in ("FEAT_DotProd", "FEAT_I8MM", "FEAT_SME2") if re.search(rf"{f}: 1", flags)]
        return name.strip(), have
    lines = Path("/proc/cpuinfo").read_text().splitlines()
    name = next((line.split(":", 1)[1] for line in lines if line.startswith(("model name", "CPU part"))), "?")
    flags = next((line.split(":", 1)[1].split() for line in lines if line.startswith(("flags", "Features"))), [])
    return name.strip(), [f for f in FEATURES if f in flags]


def predict(classifier: Classifier, texts: list[str]) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), 64):
        out.extend(np.argmax(classifier.logits(texts[i : i + 64]), axis=-1).tolist())
    return out


def main() -> None:
    name, features = cpu()
    print(name, platform.machine(), features, "onnxruntime", ort.__version__)
    data = load()
    texts, truth = data.test.texts, data.test.labels
    report: dict[str, object] = {"cpu": name, "machine": platform.machine(), "features": features}
    report["onnxruntime"] = ort.__version__
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "onnx"
        for f in ("onnx/model.onnx", "onnx/model.int8.onnx", "onnx/tokenizer.json", "onnx/labels.json"):
            hf_hub_download(REPO, f, revision=SHIPPED, local_dir=tmp)
        first = hf_hub_download(REPO, "onnx/model.int8.onnx", revision=FIRST_INT8, local_dir=Path(tmp) / "first")
        shutil.copy(first, d / "int8-8bit.onnx")
        quant_pre_process(str(d / "model.onnx"), str(d / "prepared.onnx"))
        quantize_dynamic(str(d / "prepared.onnx"), str(d / "u8u8.onnx"), weight_type=QuantType.QUInt8, per_channel=True)

        no_fusions = ort.SessionOptions()
        no_fusions.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        runs = {
            "float32": ("model.onnx", None),
            "int8_8bit": ("int8-8bit.onnx", None),
            "int8_7bit": ("model.int8.onnx", None),
            "int8_7bit_no_fusions": ("model.int8.onnx", no_fusions),
            "int8_unsigned": ("u8u8.onnx", None),
        }
        for key, (file, options) in runs.items():
            classifier = Classifier(d, file)
            if options:
                classifier.session = ort.InferenceSession(str(d / file), options, providers=["CPUExecutionProvider"])
            predicted = predict(classifier, texts)
            accuracy = sum(p == t for p, t in zip(predicted, truth, strict=True)) / len(truth)
            report[key] = {"accuracy": accuracy, "predictions": predicted}
            print(f"{key:22s} test accuracy {accuracy:.4f}")
    tag = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    out = ROOT / "results" / "serving" / f"cpu-{tag}.json"
    out.write_text(json.dumps(report) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
