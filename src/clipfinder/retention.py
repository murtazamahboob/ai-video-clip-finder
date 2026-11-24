"""Retention predictor — the blueprint's Part 3 upgrade #1.

A logistic regression trained from scratch (NumPy, gradient descent) on
the user's own labelled clips: did viewers stay past 30 s? Features are
[hook, emotion, insight, audio_energy, length_fit], each 0..1. The bundled
training set is synthetic with a *known* ground-truth rule, so
``benchmark()`` can report both accuracy and how close the fitted weights
are to the true ones — the model is checked, not just asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = ["hook", "emotion", "insight", "audio_energy", "length_fit"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class RetentionModel:
    weights: np.ndarray  # (5,)
    bias: float
    train_accuracy: float = 0.0

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return _sigmoid(x @ self.weights + self.bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)

    def feature_importance(self) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.weights))
        return [(FEATURE_NAMES[i], float(self.weights[i])) for i in order]


def train_retention(x: np.ndarray, y: np.ndarray, lr: float = 0.5,
                    epochs: int = 800, seed: int = 1) -> RetentionModel:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, x.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = _sigmoid(x @ w + b)
        grad_w = x.T @ (p - y) / len(y) + 0.01 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    acc = float((( _sigmoid(x @ w + b) >= 0.5).astype(int) == y).mean())
    return RetentionModel(w, b, acc)


def benchmark(x: np.ndarray, y: np.ndarray, true_w: np.ndarray | None = None,
              seed: int = 3) -> dict:
    """80/20 split, retrain on the 80%, report held-out accuracy."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    model = train_retention(x[idx[:cut]], y[idx[:cut]])
    held = model.predict(x[idx[cut:]])
    acc = float((held == y[idx[cut:]]).mean())
    out = {"train_accuracy": round(model.train_accuracy, 3),
           "held_out_accuracy": round(acc, 3),
           "n_train": int(cut), "n_held_out": int(len(y) - cut),
           "feature_importance": [(n, round(v, 3))
                                  for n, v in model.feature_importance()]}
    if true_w is not None:
        # Logistic weights are only identified up to scale (the data are
        # near-separable, so the fit grows in magnitude), so compare the
        # *direction*: normalised weights and cosine similarity.
        fitted = model.weights / np.linalg.norm(model.weights)
        true_n = true_w / np.linalg.norm(true_w)
        out["weight_error_vs_true"] = round(
            float(np.abs(fitted - true_n).mean()), 3)
        out["weight_cosine_vs_true"] = round(
            float(fitted @ true_n), 3)
    return out
