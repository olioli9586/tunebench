import json
from collections import Counter

import pytest

from tunebench import data
from tunebench.data import Split, carve, choose_val_rows


def test_val_rows_are_stratified_and_repeatable():
    labels = [i % 7 for i in range(700)]
    rows = choose_val_rows(labels, fraction=0.1, seed=3)
    assert rows == choose_val_rows(labels, fraction=0.1, seed=3)
    assert rows != choose_val_rows(labels, fraction=0.1, seed=4)
    assert Counter(labels[i] for i in rows) == {label: 10 for label in range(7)}


def test_carve_splits_train_into_disjoint_parts():
    full = Split([f"t{i}" for i in range(10)], list(range(10)))
    train, val = carve(full, [2, 5])
    assert val.texts == ["t2", "t5"] and val.labels == [2, 5]
    assert len(train) == 8 and not set(train.texts) & set(val.texts)


@pytest.mark.parametrize("rows", [[1, 1], [10], [-1]])
def test_carve_rejects_bad_rows(rows):
    with pytest.raises(ValueError):
        carve(Split(["a"] * 10, [0] * 10), rows)


def test_committed_split_and_labels():
    rows = json.loads(data.VAL_ROWS.read_text())
    assert len(rows) == 1001 and rows == sorted(set(rows)) and rows[-1] < 10003
    names = data.label_names()
    assert len(names) == len(set(names)) == 77


def test_download_refuses_a_file_that_does_not_match_the_pin(tmp_path):
    (tmp_path / "test.parquet").write_bytes(b"not the real file")
    with pytest.raises(ValueError, match="sha256"):
        data.download("test", tmp_path)


def test_real_data_matches_the_committed_split():
    """Downloads the two pinned parquet files (about 400 kB) on first run."""
    d = data.load()
    assert (len(d.train), len(d.val), len(d.test)) == (9002, 1001, 3080)
    assert set(Counter(d.test.labels).values()) == {40}
    assert set(d.val.labels) == set(range(77))
