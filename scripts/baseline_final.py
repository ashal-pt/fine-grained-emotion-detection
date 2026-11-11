import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from emotion_nlp.config import DATA_DIR, REPORTS_DIR, set_seed
from emotion_nlp.data import drop_leaked_train, get_label_names, load_goemotions, to_multi_hot
from emotion_nlp.evaluate import (
    apply_class_thresholds, apply_threshold, multilabel_metrics, per_class_report,
    split_half_estimate, tune_global_threshold, tune_per_class_thresholds,
)

set_seed()
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
GRID = np.arange(0.01, 0.99, 0.01)  # wide enough that optima don't sit on an edge

raw = load_goemotions()
names = get_label_names(raw)
K = len(names)
ds = drop_leaked_train(raw)
Y_train = to_multi_hot(ds["train"]["labels"], K)
Y_val = to_multi_hot(ds["validation"]["labels"], K)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
X_train = vec.fit_transform(ds["train"]["text"])  # vocabulary from train only
X_val = vec.transform(ds["validation"]["text"])

final, all_probs = {}, {}
for label, weight in [("plain", None), ("balanced", "balanced")]:
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight=weight), n_jobs=-1)
    clf.fit(X_train, Y_train)
    probs = clf.predict_proba(X_val)
    all_probs[label] = probs
    np.save(DATA_DIR / f"baseline_{label}_val_probs.npy", probs)

    g_thr, g_f1 = tune_global_threshold(Y_val, probs, grid=GRID)
    th = tune_per_class_thresholds(Y_val, probs, grid=GRID)
    on_edge = int(np.sum((th <= GRID[0] + 1e-9) | (th >= GRID[-1] - 1e-9)))
    final[label] = {
        "global_threshold": g_thr,
        "macro_f1_global": g_f1,
        "macro_f1_per_class_optimistic": multilabel_metrics(
            Y_val, apply_class_thresholds(probs, th))["macro_f1"],
        "macro_f1_per_class_honest": split_half_estimate(Y_val, probs, grid=GRID),
        "per_class_thresholds_on_grid_edge": on_edge,
    }

print(f"{'model':9s} {'global':>7s} {'(thr)':>6s} {'per-class':>10s} {'honest':>7s} {'edge':>5s}")
for label, r in final.items():
    print(f"{label:9s} {r['macro_f1_global']:7.3f} {r['global_threshold']:6.2f} "
          f"{r['macro_f1_per_class_optimistic']:10.3f} {r['macro_f1_per_class_honest']:7.3f} "
          f"{r['per_class_thresholds_on_grid_edge']:5d}")

# Per-emotion detail: balanced model with its single global threshold
thr = final["balanced"]["global_threshold"]
report = per_class_report(Y_val, apply_threshold(all_probs["balanced"], thr), names)
print(f"\nBalanced model, one global threshold = {thr:.2f}:")
print(report.round(3).to_string(index=False))

report.to_csv(REPORTS_DIR / "baseline_final_per_class.csv", index=False)
with open(REPORTS_DIR / "baseline_final.json", "w") as f:
    json.dump(final, f, indent=2)
print("\nSaved reports/baseline_final.json and reports/baseline_final_per_class.csv")