"""Simple Tkinter app for testing the GoEmotions model."""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-goemotions"

MAX_LENGTH = 128
THRESHOLD = 0.14

# Keep this order aligned with the model's 28 output positions.
LABEL_NAMES = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not (MODEL_DIR / "config.json").exists():
    raise FileNotFoundError(
        f"Could not find the trained model at:\n{MODEL_DIR}\n"
        "Run the transformer training script first."
    )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

if model.config.num_labels != len(LABEL_NAMES):
    raise ValueError(
        f"Model has {model.config.num_labels} outputs, "
        f"but LABEL_NAMES contains {len(LABEL_NAMES)} labels."
    )

model.to(DEVICE)
model.eval()


def predict_emotions(text):
    """Return emotion scores at or above the selected threshold."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        scores = torch.sigmoid(logits).cpu().tolist()

    ranked = sorted(
        zip(LABEL_NAMES, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    selected = [
        (label, score)
        for label, score in ranked
        if score >= THRESHOLD
    ]

    # Always show the top result, even if none pass the threshold.
    if not selected:
        selected = ranked[:1]

    return selected


def on_predict():
    text = input_box.get("1.0", tk.END).strip()

    if not text:
        messagebox.showinfo("Text needed", "Enter some text to analyze.")
        return

    try:
        results = predict_emotions(text)
    except Exception as exc:
        messagebox.showerror("Prediction error", str(exc))
        return

    results_box.delete(*results_box.get_children())

    for label, score in results:
        results_box.insert(
            "",
            tk.END,
            values=(label, f"{score:.3f}"),
        )

    status_var.set(
        f"Device: {DEVICE} | Showing scores >= {THRESHOLD:.2f} "
        "(or the top result if none pass)."
    )


root = tk.Tk()
root.title("GoEmotions Text Tester")
root.geometry("620x520")
root.minsize(500, 420)

main = ttk.Frame(root, padding=16)
main.pack(fill=tk.BOTH, expand=True)

ttk.Label(
    main,
    text="Fine-Grained Emotion Detection",
    font=("Segoe UI", 16, "bold"),
).pack(anchor=tk.W)

ttk.Label(
    main,
    text="Enter text, then select Analyze. Multiple emotions may be returned.",
).pack(anchor=tk.W, pady=(4, 12))

input_box = tk.Text(main, height=7, wrap=tk.WORD)
input_box.pack(fill=tk.X)
input_box.insert("1.0", "I am excited and a little nervous about starting!")

ttk.Button(
    main,
    text="Analyze text",
    command=on_predict,
).pack(anchor=tk.W, pady=12)

ttk.Label(main, text="Predicted emotions and scores:").pack(anchor=tk.W)

results_box = ttk.Treeview(
    main,
    columns=("emotion", "score"),
    show="headings",
    height=10,
)
results_box.heading("emotion", text="Emotion")
results_box.heading("score", text="Model score")
results_box.column("emotion", width=250)
results_box.column("score", width=120, anchor=tk.CENTER)
results_box.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

ttk.Label(
    main,
    text="Scores are model outputs, not reliable measures of a person's true emotions.",
    wraplength=570,
).pack(anchor=tk.W, pady=(0, 6))

status_var = tk.StringVar(
    value=f"Device: {DEVICE} | Threshold: {THRESHOLD:.2f}"
)
ttk.Label(main, textvariable=status_var).pack(anchor=tk.W)

root.mainloop()