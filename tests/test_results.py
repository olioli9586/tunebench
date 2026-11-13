import json

from tunebench.results import load_all, record


def test_record_writes_scores_details_and_predictions(tmp_path):
    row = record("run-a", "baseline", [0, 1, 1], [0, 1, 0], results_dir=tmp_path, cost_usd_per_1k=0.0)
    saved = json.loads((tmp_path / "run-a.json").read_text())
    assert saved == row
    assert saved["accuracy"] == 2 / 3 and saved["predictions"] == [0, 1, 0] and saved["n"] == 3
    assert [r["name"] for r in load_all(tmp_path)] == ["run-a"]
