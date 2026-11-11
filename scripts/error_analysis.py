"""Inspect difficult test labels and representative prediction errors."""

import numpy as np
import pandas as pd

from emotion_nlp.config import DATA_DIR, REPORTS_DIR
from emotion_nlp.data import (
    drop_leaked_train,
    get_label_names,
    load_goemotions,
    to_multi_hot,
)
from emotion_nlp.evaluate import per_class_report

NUM_DIFFICULT_LABELS = 5
EXAMPLES_PER_ERROR_TYPE = 3

saved = np.load(DATA_DIR / "test_predictions.npz")

y_true = saved["y_true"].astype(np.int8)

models = {
    "TF-IDF": {
        "probabilities": saved["baseline_probabilities"],
        "predictions": saved["baseline_predictions"].astype(np.int8),
    },
    "DistilBERT": {
        "probabilities": saved["transformer_probabilities"],
        "predictions": saved["transformer_predictions"].astype(np.int8),
    },
}

dataset = drop_leaked_train(load_goemotions())
label_names = get_label_names(dataset)
test_texts = dataset["test"]["text"]
dataset_test_labels = to_multi_hot(dataset["test"]["labels"], len(label_names))

# Make sure saved predictions still match the current test split and row order.
assert len(test_texts) == len(y_true)
assert np.array_equal(y_true, dataset_test_labels)

transformer_report = per_class_report(
    y_true,
    models["DistilBERT"]["predictions"],
    label_names,
).sort_values("f1", ascending=True)

difficult_labels = transformer_report.head(NUM_DIFFICULT_LABELS)["label"].tolist()

print("Lowest-performing DistilBERT test labels:")
print(
    transformer_report.head(NUM_DIFFICULT_LABELS)[
        ["label", "f1", "support"]
    ].round(3).to_string(index=False)
)

error_rows = []

for label in difficult_labels:
    label_index = label_names.index(label)

    for model_name, model_data in models.items():
        probabilities = model_data["probabilities"][:, label_index]
        predictions = model_data["predictions"][:, label_index]
        actual = y_true[:, label_index]

        error_groups = {
            "false_positive": np.where((actual == 0) & (predictions == 1))[0],
            "false_negative": np.where((actual == 1) & (predictions == 0))[0],
        }

        for error_type, indices in error_groups.items():
            # Show the most confident mistakes of each type.
            if error_type == "false_positive":
                ordered = indices[np.argsort(probabilities[indices])[::-1]]
            else:
                ordered = indices[np.argsort(probabilities[indices])]

            for row_index in ordered[:EXAMPLES_PER_ERROR_TYPE]:
                gold_labels = [
                    label_names[i]
                    for i in np.where(y_true[row_index] == 1)[0]
                ]

                error_rows.append(
                    {
                        "emotion_being_checked": label,
                        "model": model_name,
                        "error_type": error_type,
                        "test_row": int(row_index),
                        "probability_for_emotion": float(probabilities[row_index]),
                        "gold_labels": ", ".join(gold_labels),
                        "text": test_texts[row_index],
                    }
                )

examples = pd.DataFrame(error_rows)
output_path = REPORTS_DIR / "error_examples.csv"
examples.to_csv(output_path, index=False)

print(f"\nSaved {len(examples)} examples to: {output_path}")
print("\nExample rows:")
print(examples.head(8).to_string(index=False, max_colwidth=100))