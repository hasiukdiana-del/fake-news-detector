"""Evaluate trained models on the held-out test split.

Writes reports/metrics.json (read by the Streamlit app) and figures:
confusion matrices and ROC curves.

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import config
from .data import load_splits
from .predict import load_model, predict_proba


def compute_metrics(y_true, p_fake, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(p_fake) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_fake)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _plots(y_true, scores: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    for name, p in scores.items():
        cm = confusion_matrix(y_true, (np.asarray(p) >= 0.5).astype(int))
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=["real", "fake"], yticklabels=["real", "fake"], ax=ax)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(f"Confusion matrix: {name}")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"confusion_{name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    _, _, test = load_splits()
    y = test["label"].to_numpy()
    results, scores = {}, {}
    for name in ("baseline", "dnn", "bilstm"):
        try:
            model = load_model(name)
        except (FileNotFoundError, ImportError) as err:
            print(f"[skip] {name}: {err}")
            continue
        p = predict_proba(model, name, test["clean"].tolist(), cleaned=True)
        scores[name] = p
        results[name] = compute_metrics(y, p)
        print(name, json.dumps(results[name], indent=2))
    if not results:
        raise SystemExit("No trained models found. Run `python -m src.train` first.")
    with open(config.METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    _plots(y, scores)
    print(f"Saved metrics to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
