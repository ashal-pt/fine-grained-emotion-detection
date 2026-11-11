from collections import Counter
from datasets import load_dataset
import pandas as pd

# 1. Load the "simplified" version
ds = load_dataset("google-research-datasets/go_emotions", "simplified")
print(ds)  # shows the splits and their sizes

# 2. Column names and the 28 label names
print("Columns:", ds["train"].column_names)
label_names = ds["train"].features["labels"].feature.names
print(len(label_names), "labels:", label_names)

# 3. Look at a few raw examples (ids -> names)
for row in ds["train"].select(range(5)):
    print(row["text"], "->", [label_names[i] for i in row["labels"]])

# 4. How many labels does each comment have?
train = ds["train"].to_pandas()
train["n_labels"] = train["labels"].apply(len)
print("\nLabels per comment:\n", train["n_labels"].value_counts().sort_index())

# 5. How common is each emotion?
counts = Counter(i for labs in train["labels"] for i in labs)
freq = pd.Series({label_names[i]: c for i, c in counts.items()}).sort_values(ascending=False)
print("\nLabel frequency:\n", freq)

# 6. Text length in words
train["n_words"] = train["text"].str.split().apply(len)
print("\nWords per comment:\n", train["n_words"].describe())