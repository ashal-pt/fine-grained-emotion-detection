import numpy as np
import pytest

from emotion_nlp.evaluate import multilabel_metrics


def test_metrics_match_hand_calculation():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_pred = np.array([[1, 0], [1, 0], [1, 1]])
    m = multilabel_metrics(y_true, y_pred)
    # label 0: P=2/3, R=1   -> F1 = 0.8
    # label 1: P=1,   R=1/2 -> F1 = 2/3
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    # pooled: TP=3, FP=1, FN=1 -> P=R=0.75 -> F1 = 0.75
    assert m["micro_f1"] == pytest.approx(0.75)