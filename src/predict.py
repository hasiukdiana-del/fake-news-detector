"""Inference helpers shared by evaluation, the CLI and the Streamlit app.

Usage:
    python -m src.predict --model dnn "Some news text ..."
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from . import config
from .data import clean_text


def load_model(name: str):
    if name == "baseline":
        if not config.BASELINE_PATH.exists():
            raise FileNotFoundError(config.BASELINE_PATH)
        return joblib.load(config.BASELINE_PATH)
    if name == "dnn":
        if not config.DNN_PATH.exists():
            raise FileNotFoundError(config.DNN_PATH)
        return joblib.load(config.DNN_PATH)
    if name == "bilstm":
        if not config.BILSTM_PATH.exists():
            raise FileNotFoundError(config.BILSTM_PATH)
        from tensorflow import keras
        return keras.models.load_model(config.BILSTM_PATH)
    raise ValueError(f"Unknown model: {name}")


def predict_proba(model, name: str, texts: list[str], cleaned: bool = False) -> np.ndarray:
    """Probability that each text is fake."""
    texts = list(texts) if cleaned else [clean_text(t) for t in texts]
    if name in ("baseline", "dnn"):
        return model.predict_proba(texts)[:, 1]
    from .models import to_tensor
    return model.predict(to_tensor(texts), verbose=0).ravel()


def verdict(p_fake: float) -> str:
    if p_fake >= config.UNCERTAIN_HIGH:
        return "fake"
    if p_fake <= config.UNCERTAIN_LOW:
        return "real"
    return "uncertain"


def explain_baseline(model, text: str, top_k: int = 8) -> list[tuple[str, float]]:
    """Words that pushed the baseline towards 'fake' (+) or 'real' (-).

    Contribution = TF-IDF weight of the term in this text x model coefficient.
    """
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    vec = tfidf.transform([clean_text(text)])
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return []
    contrib = vec.toarray()[0, idx] * clf.coef_[0, idx]
    names = tfidf.get_feature_names_out()[idx]
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [(str(names[i]), float(contrib[i])) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", choices=["baseline", "dnn", "bilstm"], default="dnn")
    args = parser.parse_args()
    model = load_model(args.model)
    p = float(predict_proba(model, args.model, [args.text])[0])
    print(f"p(fake) = {p:.3f}  ->  {verdict(p)}")


if __name__ == "__main__":
    main()
