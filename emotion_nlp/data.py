"""Loading and preparing GoEmotions."""
import numpy as np
from datasets import DatasetDict, load_dataset

from emotion_nlp.config import DATASET_CONFIG, DATASET_NAME


def load_goemotions() -> DatasetDict:
    """Download (or load from cache) the simplified GoEmotions splits."""
    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def get_label_names(ds: DatasetDict) -> list[str]:
    """Return the 28 label names, in id order."""
    return ds["train"].features["labels"].feature.names


def to_multi_hot(label_lists, num_labels: int) -> np.ndarray:
    """[[15, 4], [27]] -> matrix of shape (n_comments, num_labels) with 0/1."""
    y = np.zeros((len(label_lists), num_labels), dtype=np.int8)
    for row, labels in enumerate(label_lists):
        y[row, labels] = 1
    return y

def normalize_text(text: str) -> str:
    """Used only to COMPARE texts, never to change the training data."""
    return text.strip().lower()


def drop_leaked_train(ds: DatasetDict) -> DatasetDict:
    """Remove train rows whose text also appears in validation or test.

    Validation and test are returned untouched.
    """
    held_out = {
        normalize_text(t)
        for split in ("validation", "test")
        for t in ds[split]["text"]
    }
    keep = [
        i for i, t in enumerate(ds["train"]["text"])
        if normalize_text(t) not in held_out
    ]
    return DatasetDict(
        {
            "train": ds["train"].select(keep),
            "validation": ds["validation"],
            "test": ds["test"],
        }
    )