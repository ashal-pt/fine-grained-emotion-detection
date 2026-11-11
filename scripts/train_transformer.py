"""Fine-tune DistilBERT for multi-label GoEmotions classification."""

import json

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from emotion_nlp.config import REPORTS_DIR, SEED
from emotion_nlp.data import drop_leaked_train, get_label_names, load_goemotions
from emotion_nlp.evaluate import multilabel_metrics

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 4
NUM_EPOCHS = 1
MODEL_OUTPUT_DIR = "models/distilbert-goemotions"

set_seed(SEED)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

raw_dataset = load_goemotions()
label_names = get_label_names(raw_dataset)
num_labels = len(label_names)
dataset = drop_leaked_train(raw_dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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


# Keep only model inputs and the new 28-value labels.
original_columns = dataset["train"].column_names
tokenized_dataset = dataset.map(
    prepare_batch,
    batched=True,
    remove_columns=original_columns,
)

train_data = tokenized_dataset["train"]
validation_data = tokenized_dataset["validation"]


class MultiLabelDataCollator:
    """Pad token inputs and convert labels to floating-point tensors."""

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

# Verify one batch before starting the longer run.
probe = collator([train_data[i] for i in range(BATCH_SIZE)])
print("Batch labels:", probe["labels"].dtype, probe["labels"].shape)
assert probe["labels"].dtype == torch.float32
assert probe["labels"].shape == (BATCH_SIZE, num_labels)


def compute_metrics(eval_prediction):
    """Report validation F1 at the standard 0.5 probability threshold."""
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    predictions = (probabilities >= 0.5).astype(np.int8)
    true_labels = eval_prediction.label_ids.astype(np.int8)

    return multilabel_metrics(true_labels, predictions)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    use_cpu=False,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    logging_strategy="epoch",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print(f"Training examples: {len(train_data):,}")
print(f"Validation examples: {len(validation_data):,}")
print(f"Number of labels: {num_labels}")
print(f"Training for {NUM_EPOCHS} epoch on CPU...")
print("Training device:", trainer.args.device)

trainer.train()

# Trainer reloads the best checkpoint according to validation loss.
validation_metrics = trainer.evaluate(eval_dataset=validation_data)
print("\nValidation metrics:")
print(validation_metrics)

trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

metrics_path = REPORTS_DIR / "transformer_validation.json"
with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(validation_metrics, file, indent=2)

print(f"\nSaved model to: {MODEL_OUTPUT_DIR}")
print(f"Saved validation metrics to: {metrics_path}")