"""Tune one global decision threshold using validation data only."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from emotion_nlp.config import DATA_DIR, REPORTS_DIR
from emotion_nlp.data import drop_leaked_train, get_label_names, load_goemotions
from emotion_nlp.evaluate import (
    apply_threshold,
    multilabel_metrics,
    tune_global_threshold,
)

MODEL_DIR = Path("models/distilbert-goemotions")
MAX_LENGTH = 128
GRID = np.arange(0.01, 0.99, 0.01)

raw_dataset = load_goemotions()
label_names = get_label_names(raw_dataset)
num_labels = len(label_names)
dataset = drop_leaked_train(raw_dataset)
validation = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


def prepare_batch(batch):
    """Tokenize text and convert label IDs to multi-hot vectors."""
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


validation_data = validation.map(
    prepare_batch,
    batched=True,
    remove_columns=validation.column_names,
)


class MultiLabelDataCollator:
    """Pad model inputs and ensure labels are floating-point tensors."""

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


collator = MultiLabelDataCollator(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

args = TrainingArguments(
    output_dir="reports/threshold_tuning",
    use_cpu=False,
    per_device_eval_batch_size=8,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    processing_class=tokenizer,
)

print("Predicting validation probabilities...")
prediction_output = trainer.predict(validation_data)

logits = prediction_output.predictions
if isinstance(logits, tuple):
    logits = logits[0]

probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
true_labels = prediction_output.label_ids.astype(np.int8)

best_threshold, _ = tune_global_threshold(
    true_labels,
    probabilities,
    grid=GRID,
)

default_metrics = multilabel_metrics(
    true_labels,
    apply_threshold(probabilities, 0.5),
)
tuned_metrics = multilabel_metrics(
    true_labels,
    apply_threshold(probabilities, best_threshold),
)

results = {
    "threshold_selected_on_validation": best_threshold,
    "validation_at_0.5": default_metrics,
    "validation_at_tuned_threshold": tuned_metrics,
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

np.save(DATA_DIR / "transformer_val_probs.npy", probabilities)

report_path = REPORTS_DIR / "transformer_threshold_tuning.json"
with report_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print(f"\nSelected validation threshold: {best_threshold:.2f}")
print("At threshold 0.50:", default_metrics)
print("At tuned threshold:", tuned_metrics)
print(f"Saved results to: {report_path}")