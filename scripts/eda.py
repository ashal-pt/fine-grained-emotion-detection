import matplotlib

matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emotion_nlp.config import FIGURES_DIR
from emotion_nlp.data import get_label_names, load_goemotions, to_multi_hot

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ds = load_goemotions()
names = get_label_names(ds)
train = ds["train"].to_pandas()
Y = to_multi_hot(train["labels"], len(names))

# 1. Label frequency
freq = pd.Series(Y.sum(axis=0), index=names).sort_values()
fig, ax = plt.subplots(figsize=(8, 8))
freq.plot.barh(ax=ax)
ax.set_title("Label frequency (train)")
ax.set_xlabel("Number of comments")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "label_frequency.png", dpi=150)
plt.close(fig)
print("Imbalance: most common / rarest =", int(freq.max()), "/", int(freq.min()))

# 2. Words per comment
words = train["text"].str.split().str.len()
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(words, bins=range(1, 36))
ax.set_title("Words per comment (train)")
ax.set_xlabel("Words")
ax.set_ylabel("Comments")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "words_per_comment.png", dpi=150)
plt.close(fig)

# 3. Which emotions appear together?
Yi = Y.astype(np.int32)  # int8 would overflow when counting
co = Yi.T @ Yi           # co[i, j] = number of comments with both i and j
pairs = [
    (names[i], names[j], int(co[i, j]))
    for i in range(len(names))
    for j in range(i + 1, len(names))
]
pairs.sort(key=lambda p: -p[2])
print("\nTop 10 emotion pairs that co-occur:")
for a, b, c in pairs[:10]:
    print(f"  {a} + {b}: {c}")

# Heatmap: given emotion row, how often is column emotion also present?
cond = co / np.diag(co)[:, None]
np.fill_diagonal(cond, 0)
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cond, cmap="viridis")
ax.set_xticks(range(len(names)), names, rotation=90)
ax.set_yticks(range(len(names)), names)
ax.set_title("P(column emotion | row emotion), train")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "label_cooccurrence.png", dpi=150)
plt.close(fig)
print("\nSaved 3 figures to", FIGURES_DIR)