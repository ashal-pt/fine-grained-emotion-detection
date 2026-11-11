"""Demonstrate DistilBERT tokenization on examples and GoEmotions."""

from transformers import AutoTokenizer

from emotion_nlp.data import load_goemotions

MODEL_NAME = "distilbert-base-uncased"

# Load tokenizer files. This does not load the model or require PyTorch.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Show how a few comments are split into tokens and token IDs.
examples = [
    "My favourite food is anything I didn't have to cook myself.",
    "WHY IS THIS HAPPENING AGAIN???",
    "Thank you so much, [NAME]! Unbelievably kind.",
]

for text in examples:
    encoded = tokenizer(text)
    print(text)
    print("  tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print("  ids   :", encoded["input_ids"])
    print()

# Load the dataset's training text.
dataset = load_goemotions()
raw_texts = dataset["train"]["text"]

# Ensure the tokenizer receives a list of strings, even if any value is missing.
texts = [text if isinstance(text, str) else "" for text in raw_texts]

print(f"Training comments: {len(texts):,}")
print("First text type:", type(texts[0]).__name__ if texts else "no rows")

# Tokenize in batches to avoid sending the whole dataset in one call.
lengths = []
batch_size = 1_000

for start in range(0, len(texts), batch_size):
    batch = texts[start : start + batch_size]
    encoded_batch = tokenizer(batch, truncation=False)
    lengths.extend(len(token_ids) for token_ids in encoded_batch["input_ids"])

if lengths:
    lengths_sorted = sorted(lengths)
    p95_index = int(0.95 * (len(lengths_sorted) - 1))

    print("\nToken length summary (including [CLS] and [SEP]):")
    print(f"  shortest: {min(lengths)}")
    print(f"  median:   {lengths_sorted[len(lengths_sorted) // 2]}")
    print(f"  95th percentile: {lengths_sorted[p95_index]}")
    print(f"  longest:  {max(lengths)}")