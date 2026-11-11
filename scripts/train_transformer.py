"""Run a tiny CPU training check for multi-label GoEmotions."""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from emotion_nlp.data import drop_leaked_train, get_label_names, load_goemotions

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_TRAIN_EXAMPLES = 16
NUM_EVAL_EXAMPLES = 8

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


# Drop the original text, label IDs, and dataset ID after preparing model inputs.
original_columns = dataset["train"].column_names
tokenized_dataset = dataset.map(
    prepare_batch,
    batched=True,
    remove_columns=original_columns,
)

train_data = tokenized_dataset["train"].select(range(NUM_TRAIN_EXAMPLES))
eval_data = tokenized_dataset["validation"].select(range(NUM_EVAL_EXAMPLES))


class MultiLabelDataCollator:
    """Pad token inputs and convert multi-label targets to float tensors."""

    def __init__(self, tokenizer):
        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
        }

        # Ignore metadata such as text or id if any extra column remains.
        features = [
            {key: value for key, value in feature.items() if key in allowed_keys}
            for feature in features
        ]

        batch = self.padding_collator(features)
        batch["labels"] = batch["labels"].to(dtype=torch.float32)
        return batch


collator = MultiLabelDataCollator(tokenizer)

# Confirm the labels have the right numeric type and 28-label shape.
probe = collator([train_data[i] for i in range(4)])
print("Batch keys:", list(probe.keys()))
print("Batch labels:", probe["labels"].dtype, probe["labels"].shape)

assert probe["labels"].dtype == torch.float32
assert probe["labels"].shape == (4, num_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

training_args = TrainingArguments(
    output_dir="reports/transformer_smoke",
    use_cpu=True,
    max_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=collator,
    processing_class=tokenizer,
)

print(f"Labels: {num_labels}")
print(f"Training examples: {len(train_data)}")
print("Starting 2-step smoke run...")

trainer.train()

print("\nEvaluating the tiny validation sample...")
evaluation_results = trainer.evaluate(eval_dataset=eval_data)
print(evaluation_results)