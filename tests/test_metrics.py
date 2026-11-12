import pytest

from tunebench.metrics import confusion, mcnemar, most_confused, percentiles, scores


def test_scores():
    s = scores([0, 0, 1, 1], [0, 1, 1, 1], n_labels=2)
    assert s["accuracy"] == 0.75
    # F1 for 0: 2/(2+0+1) = 2/3; for 1: 4/(4+1+0) = 0.8
    assert s["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_no_answer_is_wrong_for_every_intent():
    s = scores([0, 1], [-1, 1], n_labels=2)
    assert s["accuracy"] == 0.5
    assert s["macro_f1"] == pytest.approx((0 + 1) / 2)


def test_scores_needs_matching_lengths():
    with pytest.raises(ValueError):
        scores([0, 1], [0])


def test_confusion_skips_no_answer():
    m = confusion([0, 1, 1], [1, 1, -1], n_labels=2)
    assert m.tolist() == [[0, 1], [0, 1]]


def test_most_confused_counts_both_directions():
    names = ["a", "b", "c"]
    truth = [0, 0, 1, 2, 2]
    pred = [1, 1, 0, 0, 2]
    pairs = most_confused(truth, pred, names, k=2)
    assert pairs[0] == {"a": "a", "b": "b", "errors": 3, "a_as_b": 2, "b_as_a": 1}
    assert pairs[1]["errors"] == 1 and {pairs[1]["a"], pairs[1]["b"]} == {"a", "c"}


def test_mcnemar():
    assert mcnemar([True, False], [True, False]) == 1.0
    # 10 discordant pairs, all favoring a: 2 * 0.5**10
    assert mcnemar([True] * 10, [False] * 10) == pytest.approx(2 / 1024)
    assert mcnemar([True] * 3 + [False] * 3, [False] * 3 + [True] * 3) == 1.0


def test_percentiles():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["p95"] == pytest.approx(95.05)
