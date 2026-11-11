import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from emotion_nlp.config import REPORTS_DIR, set_seed
from emotion_nlp.data import drop_leaked_train, get_label_names, load_goemotions, to_multi_hot
from emotion_nlp.evaluate import (
    apply_class_thresholds, apply_threshold, multilabel_metrics,
    split_half_estimate, tune_global_threshold, tune_per_class_thresholds,
)

set_seed()
raw = load_goemotions()
names = get_label_names(raw)
K = len(names)
ds = drop_leaked_train(raw)
Y_train = to_multi_hot(ds["train"]["labels"], K)
Y_val = to_multi_hot(ds["validation"]["labels"], K)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
X_train = vec.fit_transform(ds["train"]["text"])  # vocabulary from train only
X_val = vec.transform(ds["validation"]["text"])

results, thresholds_by_model = {}, {}
for label, weight in [("plain", None), ("balanced", "balanced")]:
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight=weight), n_jobs=-1)
    clf.fit(X_train, Y_train)
    probs = clf.predict_proba(X_val)

    g_thr, g_f1 = tune_global_threshold(Y_val, probs)
    th = tune_per_class_thresholds(Y_val, probs)
    results[label] = {
        "macro_f1_at_0.5": multilabel_metrics(Y_val, apply_threshold(probs, 0.5))["macro_f1"],
        "best_global_threshold": g_thr,
        "macro_f1_global_tuned": g_f1,
        "macro_f1_per_class_optimistic": multilabel_metrics(
            Y_val, apply_class_thresholds(probs, th))["macro_f1"],
        "macro_f1_per_class_split_half": split_half_estimate(Y_val, probs),
    }
    thresholds_by_model[label] = th

print(f"{'model':10s} {'@0.5':>6s} {'global':>8s} {'(thr)':>6s} {'per-class':>10s} {'honest':>7s}")
for label, r in results.items():
    print(f"{label:10s} {r['macro_f1_at_0.5']:6.3f} {r['macro_f1_global_tuned']:8.3f} "
          f"{r['best_global_threshold']:6.2f} {r['macro_f1_per_class_optimistic']:10.3f} "
          f"{r['macro_f1_per_class_split_half']:7.3f}")

best = max(results, key=lambda m: results[m]["macro_f1_per_class_split_half"])
print(f"\nPer-emotion thresholds for '{best}' (tuned on validation):")
order = np.argsort(thresholds_by_model[best])
for k in order:
    print(f"  {names[k]:15s} {thresholds_by_model[best][k]:.2f}")

with open(REPORTS_DIR / "baseline_experiments.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved reports/baseline_experiments.json")