"""Compare TF-IDF and DistilBERT once on the held-out test split."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from emotion_nlp.config import DATA_DIR, REPORTS_DIR, set_seed
from emotion_nlp.data import (
    drop_leaked_train,
    get_label_names,
    load_goemotions,
    to_multi_hot,
)
from emotion_nlp.evaluate import (
    apply_threshold,
    multilabel_metrics,
    per_class_report,
)

MODEL_DIR = Path("models/distilbert-goemotions")
MAX_LENGTH = 128
EVAL_BATCH_SIZE = 8

set_seed()
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Read thresholds chosen on validation, not test.
with (REPORTS_DIR / "baseline_final.json").open(encoding="utf-8") as file:
    baseline_validation_results = json.load(file)

with (REPORTS_DIR / "transformer_threshold_tuning.json").open(
    encoding="utf-8"
) as file:
    transformer_validation_results = json.load(file)

baseline_threshold = baseline_validation_results["balanced"]["global_threshold"]
transformer_threshold = transformer_validation_results[
    "threshold_selected_on_validation"
]

raw_dataset = load_goemotions()
label_names = get_label_names(raw_dataset)
num_labels = len(label_names)
dataset = drop_leaked_train(raw_dataset)

train_split = dataset["train"]
test_split = dataset["test"]

y_train = to_multi_hot(train_split["labels"], num_labels)
y_test = to_multi_hot(test_split["labels"], num_labels)

# ---- 1. Refit the TF-IDF baseline using training data only ----
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    lowercase=True,
)

X_train = vectorizer.fit_transform(train_split["text"])
X_test = vectorizer.transform(test_split["text"])

baseline_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    n_jobs=-1,
)
baseline_model.fit(X_train, y_train)

baseline_probabilities = baseline_model.predict_proba(X_test)
baseline_predictions = apply_threshold(
    baseline_probabilities,
    baseline_threshold,
)
baseline_metrics = multilabel_metrics(y_test, baseline_predictions)

# ---- 2. Load the saved transformer and predict on test ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


def prepare_test_batch(batch):
    """Tokenize test text and convert label IDs to multi-hot vectors."""
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    multi_hot_labels = []
    for label_ids in batch["labels"]:
        row = [0.0] * num_labels
        for label_id in label_ids:
            row[label_id] = 1.0
        multi_hot_labels.append(row)

    tokenized["labels"] = multi_hot_labels
    return tokenized


test_data = test_split.map(
    prepare_test_batch,
    batched=True,
    remove_columns=test_split.column_names,
)


class MultiLabelDataCollator:
    """Pad inputs and ensure labels are floating-point tensors."""

    def __init__(self, tokenizer):
        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
        }
        features = [
            {key: value for key, value in item.items() if key in allowed_keys}
            for item in features
        ]

        batch = self.padding_collator(features)
        batch["labels"] = batch["labels"].to(dtype=torch.float32)
        return batch


model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

test_args = TrainingArguments(
    output_dir="reports/test_evaluation",
    use_cpu=False,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=test_args,
    data_collator=MultiLabelDataCollator(tokenizer),
    processing_class=tokenizer,
)

print("Predicting transformer results on the held-out test split...")
transformer_output = trainer.predict(test_data)

logits = transformer_output.predictions
if isinstance(logits, tuple):
    logits = logits[0]

transformer_probabilities = 1.0 / (
    1.0 + np.exp(-np.clip(logits, -40, 40))
)
transformer_test_labels = transformer_output.label_ids.astype(np.int8)

# Confirm the tokenizer pipeline kept test examples in the original order.
assert np.array_equal(y_test, transformer_test_labels)

transformer_predictions = apply_threshold(
    transformer_probabilities,
    transformer_threshold,
)
transformer_metrics = multilabel_metrics(y_test, transformer_predictions)

# ---- 3. Save the final comparison and per-emotion reports ----
results = {
    "test_examples": len(test_split),
    "number_of_labels": num_labels,
    "baseline_tfidf_balanced": {
        "threshold_selected_on_validation": baseline_threshold,
        **baseline_metrics,
    },
    "distilbert": {
        "threshold_selected_on_validation": transformer_threshold,
        **transformer_metrics,
    },
}

with (REPORTS_DIR / "test_comparison.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(results, file, indent=2)

baseline_per_class = per_class_report(
    y_test,
    baseline_predictions,
    label_names,
)
transformer_per_class = per_class_report(
    y_test,
    transformer_predictions,
    label_names,
)

baseline_per_class.to_csv(
    REPORTS_DIR / "test_baseline_per_class.csv",
    index=False,
)
transformer_per_class.to_csv(
    REPORTS_DIR / "test_transformer_per_class.csv",
    index=False,
)

# Save predictions for later error analysis. The data/ folder is gitignored.
np.savez_compressed(
    DATA_DIR / "test_predictions.npz",
    y_true=y_test,
    baseline_probabilities=baseline_probabilities,
    baseline_predictions=baseline_predictions,
    transformer_probabilities=transformer_probabilities,
    transformer_predictions=transformer_predictions,
)

print("\nFinal test comparison:")
print(
    "TF-IDF baseline:",
    {key: round(value, 4) for key, value in baseline_metrics.items()},
)
print(
    "DistilBERT:",
    {key: round(value, 4) for key, value in transformer_metrics.items()},
)
print(
    "\nMacro-F1 difference (DistilBERT minus TF-IDF):",
    round(transformer_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4),
)
print(f"\nSaved reports in: {REPORTS_DIR}")