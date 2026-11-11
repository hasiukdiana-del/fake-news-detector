"""Train the models.

Usage:
    python -m src.train                  # baseline + DNN (default)
    python -m src.train --model dnn
    python -m src.train --model bilstm   # optional, needs TensorFlow
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import joblib
import numpy as np
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline

from . import config
from .data import load_splits
from .models import build_baseline, build_dnn_parts


def train_baseline(train, val):
    model = build_baseline()
    start = time.time()
    model.fit(train["clean"], train["label"])
    acc = (model.predict(val["clean"]) == val["label"]).mean()
    joblib.dump(model, config.BASELINE_PATH)
    print(f"[baseline] val accuracy={acc:.4f}  fit time={time.time() - start:.1f}s")
    return model


def fit_dnn(train, val, verbose: bool = True):
    """Epoch-by-epoch training with early stopping on validation log-loss.

    Returns the fitted pipeline (TF-IDF + best MLP) and the training history.
    """
    vectorizer, mlp = build_dnn_parts()
    x_train = vectorizer.fit_transform(train["clean"])
    x_val = vectorizer.transform(val["clean"])
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}
    best_loss, best_mlp, bad_epochs = np.inf, None, 0
    for epoch in range(1, config.DNN_MAX_EPOCHS + 1):
        mlp.partial_fit(x_train, y_train, classes=[0, 1])
        p_train = mlp.predict_proba(x_train)[:, 1]
        p_val = mlp.predict_proba(x_val)[:, 1]
        history["epoch"].append(epoch)
        history["train_loss"].append(float(log_loss(y_train, p_train)))
        history["val_loss"].append(float(log_loss(y_val, p_val)))
        history["val_f1"].append(float(f1_score(y_val, p_val >= 0.5)))
        if verbose:
            print(f"epoch {epoch:2d}  train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={history['val_loss'][-1]:.4f}  val_f1={history['val_f1'][-1]:.4f}")
        if history["val_loss"][-1] < best_loss - 1e-4:
            best_loss, best_mlp, bad_epochs = history["val_loss"][-1], copy.deepcopy(mlp), 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.DNN_PATIENCE:
                if verbose:
                    print(f"early stopping: best epoch {epoch - bad_epochs}")
                break
    history["best_epoch"] = int(epoch - bad_epochs)
    # Drop training-only state so the saved model stays small:
    # the vectoriser's list of pruned terms and Adam's moment estimates.
    vectorizer.stop_words_ = None
    best_mlp._optimizer = None
    best_mlp._best_coefs = best_mlp._best_intercepts = None
    return Pipeline([("tfidf", vectorizer), ("mlp", best_mlp)]), history


def train_dnn(train, val):
    model, history = fit_dnn(train, val)
    joblib.dump(model, config.DNN_PATH)
    with open(config.REPORTS_DIR / "dnn_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history)
    return model


def plot_history(history: dict, path=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    axes[0].axvline(history["best_epoch"], ls="--", color="grey", label="best epoch")
    axes[0].set_title("Log-loss")
    axes[1].plot(history["epoch"], history["val_f1"], marker="o", color="#2E6B52")
    axes[1].set_title("Validation F1 (fake)")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path or config.FIGURES_DIR / "dnn_learning_curves.png", dpi=150)
    plt.close(fig)


def train_bilstm(train, val):
    """Optional extension. Requires TensorFlow (see README)."""
    from tensorflow import keras
    from .models import build_bilstm, to_tensor

    model = build_bilstm(train["clean"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                               restore_best_weights=True)]
    model.fit(to_tensor(train["clean"]), train["label"].to_numpy(dtype=np.float32),
              validation_data=(to_tensor(val["clean"]), val["label"].to_numpy(dtype=np.float32)),
              epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=callbacks, verbose=2)
    model.save(config.BILSTM_PATH)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "dnn", "bilstm", "all"], default="all")
    args = parser.parse_args()

    config.ensure_dirs()
    train, val, _ = load_splits()
    print(f"train={len(train)}  val={len(val)}")
    if args.model in ("baseline", "all"):
        train_baseline(train, val)
    if args.model in ("dnn", "all"):
        train_dnn(train, val)
    if args.model == "bilstm":
        train_bilstm(train, val)


if __name__ == "__main__":
    main()
