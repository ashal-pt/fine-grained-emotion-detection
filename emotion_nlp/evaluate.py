"""Metrics for multi-label classification."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def apply_threshold(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Turn probabilities into 0/1 predictions."""
    return (probs >= threshold).astype(int)


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Headline numbers. macro_f1 is our main metric."""
    kw = {"zero_division": 0}
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", **kw)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", **kw)),
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> pd.DataFrame:
    """One row per emotion, sorted best to worst F1."""
    kw = {"zero_division": 0}
    return pd.DataFrame(
        {
            "label": names,
            "precision": precision_score(y_true, y_pred, average=None, **kw),
            "recall": recall_score(y_true, y_pred, average=None, **kw),
            "f1": f1_score(y_true, y_pred, average=None, **kw),
            "support": y_true.sum(axis=0),  # how many true examples
        }
    ).sort_values("f1", ascending=False)

def tune_global_threshold(y_true, probs, grid=None):
    """Best single threshold for macro-F1. Returns (threshold, macro_f1)."""
    grid = np.arange(0.02, 0.51, 0.01) if grid is None else grid
    scores = [
        multilabel_metrics(y_true, apply_threshold(probs, t))["macro_f1"] for t in grid
    ]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def tune_per_class_thresholds(y_true, probs, grid=None, default=0.5):
    """One threshold per label, each chosen to maximize that label's F1."""
    grid = np.arange(0.02, 0.51, 0.01) if grid is None else grid
    thresholds = np.full(probs.shape[1], default)
    for k in range(probs.shape[1]):
        best_f1, best_t = -1.0, default
        for t in grid:
            f1 = f1_score(y_true[:, k], (probs[:, k] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t
    return thresholds


def apply_class_thresholds(probs, thresholds):
    """Apply a different threshold to each column."""
    return (probs >= thresholds[None, :]).astype(int)


def split_half_estimate(y_true, probs, seed=42, grid=None):
    """Honest estimate of per-class tuning: tune on one half, score on the other."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_true))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    scores = []
    for fit, test in ((a, b), (b, a)):
        th = tune_per_class_thresholds(y_true[fit], probs[fit], grid=grid)
        pred = apply_class_thresholds(probs[test], th)
        scores.append(multilabel_metrics(y_true[test], pred)["macro_f1"])
    return float(np.mean(scores))