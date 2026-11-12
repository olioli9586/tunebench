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
    def __init__(self, answers):
        self.answers = answers  # message text -> intent name, or None for a refusal
        self.calls = []

    async def parse(self, **request):
        self.calls.append(request)
        text = request["messages"][0]["content"].rsplit("Message: ", 1)[1]
        intent = self.answers.get(text, "card_arrival")
        await asyncio.sleep(0)
        return SimpleNamespace(
            parsed_output=SimpleNamespace(intent=intent) if intent else None,
            stop_reason="end_turn" if intent else "refusal",
            usage=usage(),
        )


def fake_client(answers=None):
    return SimpleNamespace(messages=FakeMessages(answers or {}))


def run(client, tmp_path, **options):
    defaults = {"shots": 0, "effort": "low", "max_usd": 1.0, "log": tmp_path / "log.jsonl", "limit": None}
    return asyncio.run(claude.classify_all(client, DATA, **(defaults | options)))


def test_answers_are_intent_indices_and_a_refusal_is_minus_one(tmp_path):
    client = fake_client({"top up failed": "top_up_failed", "help": None})
    rows = run(client, tmp_path)
    assert [r["predicted"] for r in rows] == [0, 1, 0, 0, -1]
    request = client.messages.calls[0]
    assert request["model"] == claude.MODEL and request["output_config"] == {"effort": "low"}
    assert set(request["output_format"].model_json_schema()["properties"]["intent"]["enum"]) == set(NAMES)
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
