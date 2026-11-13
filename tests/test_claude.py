import asyncio
import json
from types import SimpleNamespace

import pytest

from tunebench import claude
from tunebench.data import Banking77, Split

NAMES = ["card_arrival", "top_up_failed", "lost_or_stolen_card"]
DATA = Banking77(
    train=Split(["where is my card", "top up did not work", "someone stole my card"], [0, 1, 2]),
    val=Split([], []),
    test=Split(["card not here yet", "top up failed", "I lost my card", "card?", "help"], [0, 1, 2, 0, 1]),
    names=NAMES,
)


def usage(input_tokens=500, output_tokens=100):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


class FakeMessages:
    def __init__(self, answers, fail_on=None):
        self.answers = answers  # message text -> intent name, None for a refusal, or raw text
        self.fail_on = fail_on  # message text that raises, like an API error after retries
        self.calls = []

    async def create(self, **request):
        self.calls.append(request)
        text = request["messages"][0]["content"].rsplit("Message: ", 1)[1]
        await asyncio.sleep(0)
        if text == self.fail_on:
            raise ConnectionError("overloaded")
        answer = self.answers.get(text, "card_arrival")
        if answer is None:
            return SimpleNamespace(content=[], stop_reason="refusal", usage=usage())
        body = answer if answer.startswith("{") else json.dumps({"intent": answer})
        stop = "end_turn" if body.endswith("}") else "max_tokens"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=body)], stop_reason=stop, usage=usage())


def fake_client(answers=None, fail_on=None):
    return SimpleNamespace(messages=FakeMessages(answers or {}, fail_on))


def run(client, tmp_path, **options):
    defaults = {"shots": 0, "effort": "low", "max_usd": 1.0, "log": tmp_path / "log.jsonl", "limit": None}
    return asyncio.run(claude.classify_all(client, DATA, **(defaults | options)))


def test_answers_are_intent_indices_and_a_refusal_is_minus_one(tmp_path):
    client = fake_client({"top up failed": "top_up_failed", "help": None})
    rows = run(client, tmp_path)
    assert [r["predicted"] for r in rows] == [0, 1, 0, 0, -1]
    request = client.messages.calls[0]
    assert request["model"] == claude.MODEL and request["output_config"]["effort"] == "low"
    schema = request["output_config"]["format"]["schema"]
    assert schema["properties"]["intent"]["enum"] == NAMES and schema["required"] == ["intent"]
    assert rows[0]["cost_usd"] == pytest.approx((500 * 0.10 + 100 * 0.50) / 1e6)


def test_few_shot_adds_the_most_similar_training_messages(tmp_path):
    client = fake_client()
    run(client, tmp_path, shots=1, limit=1)
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "Message: where is my card\nIntent: card_arrival" in content
    assert content.endswith("Message: card not here yet")


def test_spend_cap_stops_new_requests_and_keeps_the_answers(tmp_path):
    client = fake_client()
    cap = 2.5 * claude.WORST_CASE_USD  # room for two requests in flight, never three
    with pytest.raises(SystemExit, match="spend cap"):
        run(client, tmp_path, max_usd=cap)
    answered = len((tmp_path / "log.jsonl").read_text().splitlines())
    assert 0 < answered < 5 and len(client.messages.calls) == answered

    # A rerun with a higher cap asks only about the rest.
    again = fake_client()
    rows = run(again, tmp_path, max_usd=1.0)
    assert len(rows) == 5 and len(again.messages.calls) == 5 - answered
    assert len({json.loads(line)["i"] for line in (tmp_path / "log.jsonl").read_text().splitlines()}) == 5


def test_a_cut_off_answer_counts_as_wrong_and_its_tokens_as_spent(tmp_path):
    client = fake_client({"top up failed": '{"intent": "top_up_fa'})
    rows = run(client, tmp_path)
    assert rows[1]["predicted"] == -1 and rows[1]["stop_reason"] == "max_tokens" and rows[1]["cost_usd"] > 0


def test_a_failed_request_keeps_every_paid_answer(tmp_path):
    with pytest.raises(RuntimeError, match="1 requests failed"):
        run(fake_client(fail_on="I lost my card"), tmp_path)
    kept = {json.loads(line)["i"] for line in (tmp_path / "log.jsonl").read_text().splitlines()}
    assert kept == {0, 1, 3, 4}
    again = fake_client()
    assert len(run(again, tmp_path)) == 5 and len(again.messages.calls) == 1


def test_the_schema_matches_what_messages_parse_sends():
    """The reported runs used messages.parse with a pydantic Literal model; the request is unchanged."""
    from typing import Literal

    from anthropic.lib._parse._transform import transform_schema
    from pydantic import create_model

    answer = create_model("Answer", intent=(Literal[tuple(NAMES)], ...))
    assert claude.answer_format(NAMES)["schema"] == transform_schema(answer.model_json_schema())
