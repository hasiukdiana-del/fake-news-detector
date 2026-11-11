"""Loading, leakage-aware cleaning and splitting of the ISOT / Kaggle
"Fake and Real News" dataset (Fake.csv + True.csv).

Why the cleaning matters
------------------------
Real articles in this dataset come from Reuters and almost all of them start
with a wire-service dateline such as "WASHINGTON (Reuters) - ". Fake articles
often end with phrases like "Featured image via Getty Images". A model trained
on raw text learns these publisher fingerprints instead of the language of
the news itself (shortcut learning). We remove them before any modelling.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

# --- publisher fingerprints -------------------------------------------------
_DATELINE = re.compile(
    r"^[^()]{0,80}\(\s*(thomson\s+)?reuters(\s+foundation)?\s*\)\s*[-\u2013\u2014]?\s*", re.IGNORECASE
)
# Reuters correction notes at the very start, e.g. "(This March 22 story was corrected ...)"
_CORRECTION_NOTE = re.compile(r"^\s*\((this|in)\b[^)]{0,300}\)\s*", re.IGNORECASE)
_LEAK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bthomson reuters foundation\b",
        r"the following statements\s+were posted to the verified twitter accounts of u\.s\. president donald (j\. )?trump"
        r"[^.]*\.(\s*the opinions expressed are his own\.)?(\s*reuters has not edited the statements[^.]*\.)?",
        r"\breuters\b",
        r"featured image via[^.\n]*",
        r"(photo|image)s? (by|via|credit)[^.\n]*",
        r"getty images?",
        r"pic\.twitter\.com/\S+",
        r"21st century wire",
        r"\bvia\s*:\s*\S+",
    )
]
# Tokens found by the log-odds audit in the research notebook (section 2.3): embedded
# JavaScript of social widgets, Reuters graphics links, image credits and
# newsletter boilerplate of specific sites. They identify the source, not
# the content, so they are dropped after normalisation.
LEAK_TOKENS = frozenset(
    """
    xfbml jssdk sdk js fjs var src cdata insertbefore parentnode createelement
    getelementsbytagname getelementbyid tmsnrt flickr screenshot acr
    subscribe subscribing filessupport henningsen wikimedia gt wfb
    uninterruptible boiler hesher
    """.split()
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@\w+")
_NON_LETTERS = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def strip_leakage(text: str) -> str:
    """Remove source-identifying artefacts but keep the article's language."""
    text = _CORRECTION_NOTE.sub("", str(text))
    text = _DATELINE.sub("", text)
    for pattern in _LEAK_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def normalise(text: str) -> str:
    """Lowercase, drop URLs, handles, digits and punctuation, squeeze spaces."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    text = _NON_LETTERS.sub(" ", text)
    return " ".join(w for w in text.split() if w not in LEAK_TOKENS)


def clean_text(text: str) -> str:
    """Full cleaning used for training AND for inference in the app."""
    return normalise(strip_leakage(text))


def load_raw(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Read Fake.csv and True.csv and return one labelled frame."""
    fake_path, true_path = raw_dir / "Fake.csv", raw_dir / "True.csv"
    missing = [p.name for p in (fake_path, true_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Download the dataset from "
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )
    fake = pd.read_csv(fake_path).assign(label=config.LABEL_FAKE)
    real = pd.read_csv(true_path).assign(label=config.LABEL_REAL)
    df = pd.concat([fake, real], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model input, clean it and remove empty and duplicate rows.

    Only title + body are used. `subject` and `date` are deliberately dropped:
    their values differ by source file and would leak the label.
    """
    out = df.copy()
    out["raw"] = (out["title"] + ". " + out["text"]).str.strip()
    out["clean"] = out["raw"].map(clean_text)
    out = out[out["clean"].str.split().str.len() >= 5]
    # Duplicates must be removed BEFORE splitting, otherwise the same article
    # can appear in train and test and inflate the scores.
    out = out.drop_duplicates(subset="clean").reset_index(drop=True)
    return out[["raw", "clean", "label"]]


def split(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified 70/15/15 split."""
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=seed
    )
    val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_splits(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR):
    """Load, clean, split and save the three splits as CSV."""
    config.ensure_dirs()
    df = prepare(load_raw(raw_dir))
    train, val, test = split(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out_dir / f"{name}.csv", index=False)
    return train, val, test


def load_splits(out_dir: Path = config.PROCESSED_DIR):
    parts = []
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}.csv"
        if not path.exists():
            return build_splits(out_dir=out_dir)
        parts.append(pd.read_csv(path, keep_default_na=False))
    return tuple(parts)


if __name__ == "__main__":
    tr, va, te = build_splits()
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")
    print("fake share in train:", round(tr["label"].mean(), 3))
