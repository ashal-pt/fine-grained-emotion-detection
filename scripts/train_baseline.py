import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from emotion_nlp.config import DATA_DIR, REPORTS_DIR, set_seed
from emotion_nlp.data import drop_leaked_train, get_label_names, load_goemotions, to_multi_hot
from emotion_nlp.evaluate import apply_threshold, multilabel_metrics, per_class_report

set_seed()
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

raw = load_goemotions()
names = get_label_names(raw)
K = len(names)
ds = drop_leaked_train(raw)

train_text, val_text = ds["train"]["text"], ds["validation"]["text"]
Y_train = to_multi_hot(ds["train"]["labels"], K)
Y_val = to_multi_hot(ds["validation"]["labels"], K)

# ---- 1. The floor: always predict "neutral" ----
dummy_pred = np.zeros_like(Y_val)
dummy_pred[:, names.index("neutral")] = 1
dummy = multilabel_metrics(Y_val, dummy_pred)
print("Always-neutral floor:", {k: round(v, 3) for k, v in dummy.items()})

# ---- 2. Three TF-IDF variants ----
CONFIGS = {
    "lowercase_words": dict(lowercase=True),
    "keep_case": dict(lowercase=False),
    "keep_case_and_punct": dict(lowercase=False, token_pattern=r"\b\w+\b|[^\w\s]"),
}


def run(cfg):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, **cfg)
    X_tr = vec.fit_transform(train_text)  # learn vocabulary from train ONLY
    X_va = vec.transform(val_text)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
    clf.fit(X_tr, Y_train)
    return clf.predict_proba(X_va), X_tr.shape[1]


results, all_probs = {}, {}
for name, cfg in CONFIGS.items():
    t0 = time.time()
    probs, n_features = run(cfg)
    m = multilabel_metrics(Y_val, apply_threshold(probs, 0.5))
    m["n_features"] = n_features
    results[name], all_probs[name] = m, probs
    print(f"{name:22s} macro_f1={m['macro_f1']:.3f} micro_f1={m['micro_f1']:.3f} "
          f"features={n_features} ({time.time() - t0:.0f}s)")

best = max(results, key=lambda n: results[n]["macro_f1"])
probs = all_probs[best]
print("\nBest variant on validation:", best)

# ---- 3. A taste of why thresholds matter ----
sweep = {}
print("\nThreshold  macro_f1  micro_f1")
for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    m = multilabel_metrics(Y_val, apply_threshold(probs, t))
    sweep[str(t)] = m
    print(f"  {t:.2f}      {m['macro_f1']:.3f}     {m['micro_f1']:.3f}")

# ---- 4. Per-emotion report at threshold 0.5, plus saved outputs ----
report = per_class_report(Y_val, apply_threshold(probs, 0.5), names)
print("\nPer-emotion results (threshold 0.5):")
print(report.round(3).to_string(index=False))

report.to_csv(REPORTS_DIR / "baseline_per_class.csv", index=False)
np.save(DATA_DIR / "baseline_val_probs.npy", probs)  # for error analysis later
with open(REPORTS_DIR / "baseline_results.json", "w") as f:
    json.dump({"floor_always_neutral": dummy, "variants": results,
               "best_variant": best, "threshold_sweep": sweep}, f, indent=2)
print("\nSaved reports/baseline_results.json, reports/baseline_per_class.csv")