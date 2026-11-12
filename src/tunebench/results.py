"""One JSON file per run in results/, holding its scores and its test-set predictions.

Keeping the predictions makes every number in RESULTS.md reproducible from the
repo alone (scripts/report.py), and lets two runs be compared example by
example (McNemar) without running either again.
"""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from .metrics import scores

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def record(
    name: str,
    approach: str,
    truth: list[int],
    predicted: list[int],
    results_dir: Path = RESULTS_DIR,
    **details: object,
) -> dict[str, object]:
    """Score a run on the test set and write results/<name>.json. Returns what was written."""
    row = {
        "name": name,
        "approach": approach,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}",
        "n": len(truth),
        **scores(truth, predicted),
        **details,
        "predictions": predicted,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}.json").write_text(json.dumps(row, indent=1) + "\n")
    return row


def load_all(results_dir: Path = RESULTS_DIR) -> list[dict[str, object]]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
