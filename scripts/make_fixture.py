"""Writes tests/fixtures/tiny/: a tiny random classifier exported like the real one.

CI and the Docker smoke test serve it, so they check the whole serving path
(tokenizer.json, labels.json, int8 ONNX) without the real model.
"""

import shutil
import tempfile
from pathlib import Path

from tunebench.data import ROOT, label_names, read_split
from tunebench.export import export
from tunebench.tiny import make_tiny_model

out = ROOT / "tests" / "fixtures" / "tiny"
with tempfile.TemporaryDirectory() as tmp:
    model_dir = make_tiny_model(Path(tmp) / "hf", read_split("train").texts, label_names())
    fp32, int8 = export(model_dir, Path(tmp) / "onnx")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("model.int8.onnx", "tokenizer.json", "labels.json"):
        shutil.copy(Path(tmp) / "onnx" / name, out / name)
for p in sorted(out.iterdir()):
    print(f"{p.name}: {p.stat().st_size / 1e3:.0f} kB")
