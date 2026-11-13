"""Approach 1: TF-IDF features and logistic regression, the baseline to beat.

Word unigrams and bigrams catch the topic ("card", "top up"); character
n-grams catch spelling variants and typos ("recieved", "cancelled"). The
regularization strength is picked on validation; the model is then the one
fit on train alone, the same data the other approaches train on.
"""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import Split
from .metrics import scores

C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def make(c: float) -> Pipeline:
    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ]
    )
    return Pipeline([("tfidf", features), ("lr", LogisticRegression(C=c, max_iter=3000))])


def fit(train: Split, val: Split) -> tuple[Pipeline, dict[str, object]]:
    """Fit one model per C, keep the best on validation macro-F1."""
    tried = {}
    best = None
    for c in C_GRID:
        started = time.perf_counter()
        model = make(c).fit(train.texts, train.labels)
        seconds = time.perf_counter() - started
        val_scores = scores(val.labels, model.predict(val.texts).tolist())
        tried[c] = {**val_scores, "train_seconds": round(seconds, 1)}
        if best is None or val_scores["macro_f1"] > tried[best[0]]["macro_f1"]:
            best = (c, model)
    assert best is not None
    return best[1], {"c": best[0], "validation": tried}


def single_message_ms(model: Pipeline, texts: list[str]) -> list[float]:
    """Time to classify one message at a time, as a server would."""
    times = []
    for text in texts:
        started = time.perf_counter()
        model.predict([text])
        times.append((time.perf_counter() - started) * 1000)
    return times


def main() -> None:
    from .data import load
    from .metrics import percentiles
    from .results import record

    data = load()
    model, chosen = fit(data.train, data.val)
    started = time.perf_counter()
    predicted = model.predict(data.test.texts).tolist()
    batch_seconds = time.perf_counter() - started
    latency = percentiles(single_message_ms(model, data.test.texts[:500]))
    row = record(
        "tfidf-logreg",
        "baseline",
        data.test.labels,
        predicted,
        model="TF-IDF (word 1-2 + char 2-5) + logistic regression",
        c=chosen["c"],
        validation=chosen["validation"],
        test_batch_seconds=round(batch_seconds, 2),
        latency_ms=latency,
        cost_usd_per_1k=0.0,
    )
    print(
        f"tfidf-logreg: test accuracy {row['accuracy']:.4f}, macro-F1 {row['macro_f1']:.4f}, "
        f"C={chosen['c']}, p50 {latency['p50']:.2f} ms"
    )


if __name__ == "__main__":
    main()
