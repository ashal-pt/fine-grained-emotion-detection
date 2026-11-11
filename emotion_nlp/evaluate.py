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