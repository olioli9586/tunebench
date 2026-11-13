"""Export checks that need PyTorch: skipped in CI, which installs only the serving stack."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tunebench.export import export  # noqa: E402
from tunebench.serve import Classifier  # noqa: E402
from tunebench.tiny import make_tiny_model  # noqa: E402

NAMES = [f"intent_{i}" for i in range(77)]
TEXTS = ["my card has not arrived", "top up failed", "how do I change my pin"]


def test_onnx_matches_pytorch_and_int8_stays_close(tmp_path):
    model_dir = make_tiny_model(tmp_path / "hf", TEXTS * 3, NAMES)
    fp32, int8 = export(model_dir, tmp_path / "onnx")

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    with torch.no_grad():
        expected = model(**hf_tokenizer(TEXTS, padding=True, return_tensors="pt")).logits.numpy()

    onnx = Classifier(fp32.parent, fp32.name)
    assert [e.ids for e in onnx.tokenizer.encode_batch(TEXTS)] == hf_tokenizer(TEXTS, padding=True)["input_ids"]
    np.testing.assert_allclose(onnx.logits(TEXTS), expected, atol=1e-4)
    quantized = Classifier(int8.parent, int8.name).logits(TEXTS)
    assert quantized.shape == expected.shape and np.abs(quantized - expected).max() < 0.1
    assert onnx.labels == NAMES
