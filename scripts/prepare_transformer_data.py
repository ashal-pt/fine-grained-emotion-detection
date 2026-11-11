"""Tokenize GoEmotions and convert labels to multi-hot vectors."""

from transformers import AutoTokenizer

from emotion_nlp.data import drop_leaked_train, get_label_names, load_goemotions

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 1_000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

raw_dataset = load_goemotions()
label_names = get_label_names(raw_dataset)
num_labels = len(label_names)

# Remove training comments that also occur in validation or test.
dataset = drop_leaked_train(raw_dataset)


def prepare_batch(batch):
    """Tokenize text and turn each list of label IDs into a multi-hot vector."""
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


# Process all splits in batches. Remove the original integer-ID labels,
# since we replace them with 28-value vectors.
tokenized_dataset = dataset.map(
    prepare_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["labels"],
)

# Check that the result has the shape the model will need.
for split_name in ("train", "validation", "test"):
    example = tokenized_dataset[split_name][0]

    print(f"\n{split_name}: {len(tokenized_dataset[split_name]):,} comments")
    print("  input_ids length:", len(example["input_ids"]))
    print("  label count:", len(example["labels"]))
    print("  active labels:", sum(example["labels"]))
    print("  first labels:", example["labels"][:5])

assert len(tokenized_dataset["train"][0]["labels"]) == num_labels
assert set(tokenized_dataset["train"][0]["labels"]).issubset({0.0, 1.0})

print(f"\nPrepared successfully: {num_labels} labels")