"""Approach 4: Claude Haiku 5.5 through the API, zero-shot and few-shot.

    uv run --env-file <file with ANTHROPIC_API_KEY> python -m tunebench.claude --shots 0
    uv run --env-file ... python -m tunebench.claude --shots 10 --max-usd 1.5

Zero-shot: the 77 intent names and the message. Few-shot adds the k training
messages most similar to this one (TF-IDF cosine), with their intents. The
answer is a structured output whose only field is an enum of the 77 names, so
it is always a valid intent or nothing (a refusal or a cut-off answer counts
as wrong).

Spending is capped: the run stops starting requests once the cost so far,
plus what the requests in flight could add, would pass --max-usd. Every
answer is appended to data/claude/<name>.jsonl as it arrives, so a rerun
picks up where a stopped one left off without paying twice.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import create_model

from .data import DATA_DIR, Banking77, load
from .metrics import percentiles
from .results import record

MODEL = "claude-haiku-5-5"
# USD per million tokens (platform.claude.com pricing, checked 2026-11-08 for DealScout).
INPUT_USD, OUTPUT_USD = 0.10, 0.50
MAX_TOKENS = 4096  # thinking counts toward it
CONCURRENCY = 8
WORST_CASE_USD = (2_000 * INPUT_USD + MAX_TOKENS * OUTPUT_USD) / 1e6  # one request's ceiling

SYSTEM = """You classify messages sent to a bank's customer support into exactly one of 77 intents.

Intents:
{intents}

Pick the single intent that best matches what the customer wants. Answer with the intent name only."""


def cost(usage) -> float:
    prompt = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    return (prompt * INPUT_USD + usage.output_tokens * OUTPUT_USD) / 1e6


class Neighbors:
    """The training messages most similar to a query, for few-shot examples."""

    def __init__(self, data: Banking77):
        from .baseline import make

        self.vectorizer = make(1.0).named_steps["tfidf"].fit(data.train.texts)
        self.matrix = self.vectorizer.transform(data.train.texts)
        self.train = data.train

    def __call__(self, text: str, k: int) -> list[tuple[str, int]]:
        similarity = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        top = np.argsort(-similarity, kind="stable")[:k]
        return [(self.train.texts[i], self.train.labels[i]) for i in top]


def prompt(text: str, examples: list[tuple[str, int]], names: list[str]) -> str:
    if not examples:
        return f"Message: {text}"
    shown = "\n\n".join(f"Message: {t}\nIntent: {names[label]}" for t, label in examples)
    return f"Labeled examples of similar messages:\n\n{shown}\n\nNow classify this one.\n\nMessage: {text}"


async def classify_all(
    client, data: Banking77, shots: int, effort: str, max_usd: float, log: Path, limit: int | None
) -> list[dict]:
    names = data.names
    answer = create_model("Answer", intent=(Literal[tuple(names)], ...))
    system = SYSTEM.format(intents="\n".join(names))
    neighbors = Neighbors(data) if shots else None
    texts = data.test.texts[:limit] if limit else data.test.texts

    done: dict[int, dict] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            done[row["i"]] = row
    spent = sum(r["cost_usd"] for r in done.values())
    in_flight = 0
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    log.parent.mkdir(parents=True, exist_ok=True)

    async def one(i: int) -> None:
        nonlocal spent, in_flight
        async with semaphore:
            async with lock:
                if spent + (in_flight + 1) * WORST_CASE_USD > max_usd:
                    return
                in_flight += 1
            examples = neighbors(texts[i], shots) if neighbors else []
            started = time.perf_counter()
            try:
                response = await client.messages.parse(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": prompt(texts[i], examples, names)}],
                    output_config={"effort": effort},
                    output_format=answer,
                )
            finally:
                async with lock:
                    in_flight -= 1
            parsed = response.parsed_output
            row = {
                "i": i,
                "predicted": names.index(parsed.intent) if parsed else -1,
                "stop_reason": response.stop_reason,
                "ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": cost(response.usage),
            }
            async with lock:
                spent += row["cost_usd"]
                done[i] = row
                with log.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    await asyncio.gather(*(one(i) for i in range(len(texts)) if i not in done))
    if len(done) < len(texts):
        raise SystemExit(
            f"Stopped at the spend cap: {len(done)}/{len(texts)} answered, ${spent:.3f} spent. "
            f"Raise --max-usd and rerun to finish (answers so far are kept in {log})."
        )
    return [done[i] for i in range(len(texts))]


def main() -> None:
    from anthropic import AsyncAnthropic

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", type=int, default=0, help="similar training examples per message")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--max-usd", type=float, default=1.5, help="hard cap for this run")
    parser.add_argument("--limit", type=int, help="first N test messages only (trial runs)")
    a = parser.parse_args()
    name = f"claude-haiku-5-5-{a.shots}shot-{a.effort}" + (f"-first{a.limit}" if a.limit else "")
    data = load()
    log = DATA_DIR / "claude" / f"{name}.jsonl"
    started = time.perf_counter()
    rows = asyncio.run(classify_all(AsyncAnthropic(max_retries=4), data, a.shots, a.effort, a.max_usd, log, a.limit))
    seconds = time.perf_counter() - started
    total = sum(r["cost_usd"] for r in rows)
    truth = data.test.labels[: len(rows)]
    result = record(
        name,
        "claude",
        truth,
        [r["predicted"] for r in rows],
        **({"results_dir": DATA_DIR / "claude"} if a.limit else {}),
        model=MODEL,
        shots=a.shots,
        effort=a.effort,
        cost_usd=round(total, 4),
        cost_usd_per_1k=round(total / len(rows) * 1000, 4),
        latency_ms=percentiles([r["ms"] for r in rows]),
        mean_output_tokens=round(sum(r["output_tokens"] for r in rows) / len(rows)),
        mean_input_tokens=round(sum(r["input_tokens"] for r in rows) / len(rows)),
        no_answer=sum(r["predicted"] < 0 for r in rows),
        concurrency=CONCURRENCY,
        wall_seconds=round(seconds),
    )
    print(
        f"{name}: accuracy {result['accuracy']:.4f}, macro-F1 {result['macro_f1']:.4f}, "
        f"${total:.3f} (${result['cost_usd_per_1k']:.3f} per 1K), p50 {result['latency_ms']['p50']:.0f} ms"
    )


if __name__ == "__main__":
    main()
