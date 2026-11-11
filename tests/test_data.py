import pandas as pd

from src import config
from src.data import clean_text, prepare, split, strip_leakage


def test_reuters_dateline_removed():
    raw = "WASHINGTON (Reuters) - The Senate voted on Tuesday."
    assert "reuters" not in clean_text(raw)
    assert clean_text(raw).startswith("the senate voted")


def test_reuters_variants_removed():
    note = "(This March 22 story was corrected to remove a reference) WASHINGTON (Reuters) - Text here."
    assert clean_text(note) == "text here"
    assert "thomson" not in clean_text("LONDON (Thomson Reuters Foundation) - Aid groups said.")
    boiler = ("The following statements\xa0were posted to the verified Twitter accounts of U.S. "
              "President Donald Trump, @realDonaldTrump and @POTUS. The opinions expressed are his own. "
              "Reuters has not edited the statements or confirmed their accuracy. Next sentence.")
    assert clean_text(boiler) == "next sentence"


def test_image_credits_removed():
    raw = "Trump said something. Featured image via Getty Images"
    cleaned = clean_text(raw)
    assert "getty" not in cleaned and "featured" not in cleaned


def test_urls_handles_digits_removed():
    cleaned = clean_text("See https://t.co/abc @user 2017 now!")
    assert cleaned == "see now"


def test_strip_leakage_keeps_regular_parentheses():
    assert "(AP)" not in strip_leakage("Text") and "(left)" in strip_leakage("a (left) b")


def _toy_frame(n=200):
    import random
    rng = random.Random(0)
    vocab = ["policy", "economy", "senate", "vote", "market", "health", "school", "border",
             "court", "budget", "energy", "trade", "media", "election", "climate", "union"]
    rows = []
    for i in range(n):
        label = i % 2
        body = " ".join(rng.choice(vocab) for _ in range(12))  # digits would be stripped
        rows.append({"title": f"title {i}", "text": body, "label": label})
    return pd.DataFrame(rows)


def test_prepare_drops_duplicates_and_short_texts():
    df = _toy_frame(20)
    df = pd.concat([df, df.iloc[[0]], pd.DataFrame([{"title": "", "text": "too short", "label": 1}])])
    out = prepare(df)
    assert len(out) == 20
    assert set(out.columns) == {"raw", "clean", "label"}


def test_split_is_stratified_and_disjoint():
    df = prepare(_toy_frame(400))
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert not set(train["clean"]) & set(test["clean"])
    for part in (train, val, test):
        assert abs(part["label"].mean() - 0.5) < 0.05
    assert abs(len(test) / len(df) - config.TEST_SIZE) < 0.02
