from tunebench import baseline
from tunebench.data import Split


def test_fit_picks_c_on_validation_and_classifies():
    texts = ["card has not arrived", "where is my card", "top up failed", "my top up did not work"] * 5
    labels = [0, 0, 1, 1] * 5
    split = Split(texts, labels)
    model, chosen = baseline.fit(split, split)
    assert chosen["c"] in baseline.C_GRID and set(chosen["validation"]) == set(baseline.C_GRID)
    assert model.predict(["my card still has not arrived", "the top up failed again"]).tolist() == [0, 1]
    assert len(baseline.single_message_ms(model, ["card"])) == 1
