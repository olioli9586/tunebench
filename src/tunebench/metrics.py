"""Scores shared by every approach, so the results table compares like with like."""

import math
from collections import Counter

import numpy as np


def scores(truth: list[int], predicted: list[int], n_labels: int = 77) -> dict[str, float]:
    """Accuracy and macro-F1 (the mean of the per-intent F1 scores, each intent weighted
    equally). A prediction of -1 (no valid answer) counts as wrong for every intent."""
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Need one prediction per example, and at least one example.")
    t = np.asarray(truth)
    p = np.asarray(predicted)
    f1s = []
    for label in range(n_labels):
        tp = int(np.sum((p == label) & (t == label)))
        fp = int(np.sum((p == label) & (t != label)))
        fn = int(np.sum((p != label) & (t == label)))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"accuracy": float(np.mean(t == p)), "macro_f1": float(np.mean(f1s))}


def confusion(truth: list[int], predicted: list[int], n_labels: int = 77) -> np.ndarray:
    """Rows are the true intent, columns the predicted one. Invalid answers (-1) are left out."""
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(truth, predicted, strict=True):
        if p >= 0:
            matrix[t, p] += 1
    return matrix


def most_confused(truth: list[int], predicted: list[int], names: list[str], k: int = 10) -> list[dict[str, object]]:
    """The k intent pairs mixed up most often, counting both directions together."""
    pairs: Counter[tuple[int, int]] = Counter()
    one_way: Counter[tuple[int, int]] = Counter()
    for t, p in zip(truth, predicted, strict=True):
        if p >= 0 and p != t:
            pairs[(min(t, p), max(t, p))] += 1
            one_way[(t, p)] += 1
    return [
        {
            "a": names[a],
            "b": names[b],
            "errors": count,
            "a_as_b": one_way[(a, b)],
            "b_as_a": one_way[(b, a)],
        }
        for (a, b), count in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    ]


def mcnemar(a_right: list[bool], b_right: list[bool]) -> float:
    """Exact two-sided McNemar p-value: do two models on the same examples differ in accuracy?
    Only the examples where exactly one of them is right carry information."""
    b = sum(1 for x, y in zip(a_right, b_right, strict=True) if x and not y)
    c = sum(1 for x, y in zip(a_right, b_right, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def percentiles(values_ms: list[float]) -> dict[str, float]:
    v = np.asarray(values_ms)
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
