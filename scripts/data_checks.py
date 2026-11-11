from emotion_nlp.data import load_goemotions

ds = load_goemotions()
frames = {split: ds[split].to_pandas() for split in ds}

# Normalize text only for comparing (we do NOT change the real data)
norm = {s: df["text"].str.strip().str.lower() for s, df in frames.items()}

for split, df in frames.items():
    print(f"\n[{split}] rows = {len(df)}")
    print("  missing values      :", int(df.isna().sum().sum()))
    print("  empty texts         :", int((norm[split] == "").sum()))
    print("  rows with no labels :", int((df["labels"].apply(len) == 0).sum()))
    print("  duplicate texts     :", int(norm[split].duplicated().sum()))

# Leakage: the same text in two different splits
sets = {s: set(n) for s, n in norm.items()}
print("\nLeakage check (texts shared between splits)")
print("  train and validation:", len(sets["train"] & sets["validation"]))
print("  train and test      :", len(sets["train"] & sets["test"]))
print("  validation and test :", len(sets["validation"] & sets["test"]))

# Anonymization placeholders in the text
for token in ["[NAME]", "[RELIGION]"]:
    n = int(frames["train"]["text"].str.contains(token, regex=False).sum())
    print(f"\nTrain comments containing {token}: {n}")