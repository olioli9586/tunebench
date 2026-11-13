from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tunebench.serve import Classifier, create_app

FIXTURE = Path(__file__).parent / "fixtures" / "tiny"


@pytest.fixture(scope="module")
def classifier():
    return Classifier(FIXTURE)


def test_classifier_returns_a_valid_intent(classifier):
    result = classifier.predict("my card has not arrived")
    assert result["intent"] in classifier.labels
    assert [t["intent"] for t in result["top"]][0] == result["intent"]
    assert 0 < result["confidence"] <= 1


def test_batches_of_different_lengths_are_padded(classifier):
    logits = classifier.logits(["hi", "a much longer message about a card payment that was declined"])
    assert logits.shape == (2, 77)


def test_long_input_is_truncated(classifier):
    assert classifier.logits(["card " * 500]).shape == (1, 77)


def test_api(classifier):
    client = TestClient(create_app(classifier))
    assert client.get("/health").json() == {"status": "ok", "model": "model.int8.onnx", "labels": 77}
    response = client.post("/predict", json={"text": "how do I top up?"})
    assert response.status_code == 200 and response.json()["intent"] in classifier.labels
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 1001}).status_code == 422
    assert "<textarea" in client.get("/").text
