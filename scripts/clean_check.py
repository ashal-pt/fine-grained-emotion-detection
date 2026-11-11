from collections import Counter, defaultdict

from emotion_nlp.data import (
    drop_leaked_train, get_label_names, load_goemotions, normalize_text,
)

ds = load_goemotions()
names = get_label_names(ds)

# --- 1. Remove leakage ---
clean = drop_leaked_train(ds)
print("Train rows before:", len(ds["train"]))
print("Train rows after :", len(clean["train"]))
print("Rows removed     :", len(ds["train"]) - len(clean["train"]))

train_texts = {normalize_text(t) for t in clean["train"]["text"]}
for split in ("validation", "test"):
    shared = train_texts & {normalize_text(t) for t in clean[split]["text"]}
    print(f"Shared texts, train vs {split}: {len(shared)}")

# --- 2. Do repeated texts in train always carry the same labels? ---
groups = defaultdict(list)
for text, labels in zip(clean["train"]["text"], clean["train"]["labels"]):
    groups[normalize_text(text)].append(tuple(sorted(labels)))

repeated = {t: ls for t, ls in groups.items() if len(ls) > 1}
disagree = {t: ls for t, ls in repeated.items() if len(set(ls)) > 1}
print(f"\nTexts appearing more than once in train: {len(repeated)}")
print(f"  ...of which labels disagree            : {len(disagree)}")

print("\nExamples of repeated texts and their label sets:")
for text, ls in list(repeated.items())[:8]:
    readable = [[names[i] for i in labs] for labs in ls]
    print(f"  {text!r} -> {readable}")