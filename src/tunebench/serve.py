"""The fine-tuned classifier behind an HTTP API, with ONNX Runtime on the CPU.

    MODEL_DIR=artifacts/onnx/modernbert-seed1 uvicorn tunebench.serve:app

MODEL_DIR holds what tunebench.export writes (tokenizer.json, labels.json and
the .onnx files); MODEL_FILE picks the file, model.int8.onnx by default.
This module needs only numpy, tokenizers, onnxruntime and FastAPI, so the
serving image stays free of PyTorch.

    POST /predict {"text": "..."}  ->  {"intent": ..., "confidence": ..., "top": [...]}
    GET  /health
    GET  /              a small page to try it
"""

import json
import os
from pathlib import Path

import numpy as np

MAX_LENGTH = 128
MAX_CHARS = 1000


class Classifier:
    def __init__(self, model_dir: Path, model_file: str = "model.int8.onnx", threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_LENGTH)
        if self.tokenizer.padding is None:  # export saves the model's own pad token
            self.tokenizer.enable_padding()
        self.labels: list[str] = json.loads((model_dir / "labels.json").read_text())
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / model_file), options, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.model_file = model_file

    def logits(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        arrays = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
        }
        return self.session.run(["logits"], {name: arrays[name] for name in self.inputs})[0]

    def predict(self, text: str, top: int = 3) -> dict[str, object]:
        row = self.logits([text])[0]
        p = np.exp(row - row.max())
        p /= p.sum()
        order = np.argsort(-p)[:top]
        return {
            "intent": self.labels[order[0]],
            "confidence": round(float(p[order[0]]), 4),
            "top": [{"intent": self.labels[i], "confidence": round(float(p[i]), 4)} for i in order],
        }


def create_app(classifier: Classifier):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    class Query(BaseModel):
        text: str = Field(min_length=1, max_length=MAX_CHARS)

    app = FastAPI(title="TuneBench intent classifier")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model": classifier.model_file, "labels": len(classifier.labels)}

    @app.post("/predict")
    def predict(query: Query) -> dict[str, object]:
        return classifier.predict(query.text)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE

    return app


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Banking intent classifier</title>
<style>
 body{font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:2rem auto;padding:0 1rem;color:#1d2433;background:#fbfbf8}
 textarea{width:100%;box-sizing:border-box;font:inherit;padding:.6rem;border:1px solid #c9ccd3;border-radius:6px}
 button{margin-top:.5rem;font:inherit;padding:.45rem 1rem;border:0;border-radius:6px;background:#1d4ed8;color:#fff;cursor:pointer}
 li{margin:.2rem 0} code{background:#eef0f4;padding:0 .3rem;border-radius:4px} .muted{color:#5b6475;font-size:.9rem}
 @media (prefers-color-scheme: dark){body{background:#14161b;color:#e6e8ec}textarea{background:#1d2027;color:inherit;border-color:#3a3f4b}code{background:#262a33}.muted{color:#9aa3b2}}
</style></head><body>
<h1>Banking intent classifier</h1>
<p class="muted">ModernBERT-base fine-tuned on Banking77 (77 intents), served as int8 ONNX on a CPU.
Type a message a bank customer might send.</p>
<textarea id="text" rows="3">I still haven't received my new card, it's been two weeks</textarea>
<button id="go">Classify</button>
<ol id="out"></ol>
<script>
const out = document.getElementById("out");
document.getElementById("go").onclick = async () => {
  out.textContent = "";
  const res = await fetch("predict", {method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({text: document.getElementById("text").value})});
  const body = await res.json();
  if (!res.ok) { out.textContent = "Error: " + JSON.stringify(body.detail); return; }
  for (const t of body.top) {
    const li = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = t.intent;
    li.append(code, ` ${(t.confidence * 100).toFixed(1)}%`);
    out.append(li);
  }
};
</script></body></html>
"""


def _from_env():
    model_dir = Path(os.environ.get("MODEL_DIR", "/model"))
    threads = int(os.environ["THREADS"]) if os.environ.get("THREADS") else None
    return create_app(Classifier(model_dir, os.environ.get("MODEL_FILE", "model.int8.onnx"), threads))


def __getattr__(name: str):
    # `uvicorn tunebench.serve:app` builds the app on first access, so importing
    # this module (tests, the export script) loads no model.
    if name == "app":
        return _from_env()
    raise AttributeError(name)
