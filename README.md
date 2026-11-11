# Fine-Grained Emotion Detection

A multi-label NLP project that predicts emotions in short text. It compares a TF-IDF and logistic regression baseline with a fine-tuned DistilBERT model, then provides a small Tkinter app for trying the trained model.

## Project goal

Given a short English comment, predict one or more of 28 emotion labels. Since several emotions can apply to one comment, this is a **multi-label classification** task.

The project uses the simplified [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions), which contains Reddit comments annotated with emotion labels. The model predicts patterns represented in those annotations; it does not determine a person's true emotional state.

## Results

Both systems were evaluated on the same held-out test split. Each model's decision threshold was selected using validation data only.

| Model | Threshold | Macro F1 | Micro F1 | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|
| TF-IDF + balanced logistic regression | 0.59 | 0.4532 | 0.5117 | 0.4171 | 0.5103 |
| DistilBERT | 0.14 | **0.4720** | **0.5654** | **0.4218** | **0.5873** |

DistilBERT improved test macro F1 by 0.0187 and micro F1 by 0.0537 over the baseline. The lower DistilBERT threshold increased recall, which matters when a comment can have several labels.

Detailed metrics are in [`reports/test_comparison.json`](reports/test_comparison.json). Per-emotion results are in [`reports/test_baseline_per_class.csv`](reports/test_baseline_per_class.csv) and [`reports/test_transformer_per_class.csv`](reports/test_transformer_per_class.csv).

## Methods

### Baseline

- TF-IDF word unigrams and bigrams
- One-vs-rest logistic regression
- Balanced class weights
- One global probability threshold selected on validation data

### Transformer

- `distilbert-base-uncased`
- 28-output sequence classification head
- Multi-label loss
- Maximum input length: 128 tokens
- One training epoch
- Batch size: 4
- One global probability threshold selected on validation data

The training script uses PyTorch and Hugging Face Transformers. It selects the available GPU when configured to do so; otherwise, training can run on CPU, more slowly.

## Project structure

```text
emotion_nlp/
  config.py
  data.py
  evaluate.py
scripts/
  train_baseline.py
  baseline_final.py
  train_transformer.py
  tune_transformer_threshold.py
  evaluate_test.py
  error_analysis.py
  app.py
reports/
  figures/
  baseline and transformer metrics
tests/
data/                 # ignored by Git
models/               # ignored by Git
```

## Setup

Use Python 3.10 or later and create a virtual environment from the project root.

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Install PyTorch using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Choose the CPU build or the CUDA build that matches your NVIDIA GPU and driver. Then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

The first dataset and model runs download files from Hugging Face and cache them locally. You do not need an `HF_TOKEN` for the project, though authenticated access can provide higher download limits.

## Reproduce the workflow

Run these commands from the project root, in order:

```powershell
python scripts/data_checks.py
python scripts/eda.py
python scripts/train_baseline.py
python scripts/baseline_final.py
python scripts/train_transformer.py
python scripts/tune_transformer_threshold.py
python scripts/evaluate_test.py
python scripts/error_analysis.py
```

The transformer training step saves model files under `models/`. That directory is intentionally ignored by Git because model weights are large. A fresh clone must run `scripts/train_transformer.py` before the demo app can load the model.

Run the small desktop demo after training:

```powershell
python scripts/app.py
```

The app accepts text and displays emotion scores at or above the validation-selected threshold. It shows the top-scoring emotion if none pass the threshold.

Run the unit tests with:

```powershell
pytest
```

## Error analysis

The test analysis found that rare labels were difficult for this model. For example, `relief` had 11 test examples and `grief` had 6; both had a test F1 of 0. The saved [`reports/error_examples.csv`](reports/error_examples.csv) contains sample Reddit text and should remain local unless you have reviewed the dataset's terms and privacy considerations.

Use error examples to describe model behavior, not to tune against the test set. Use validation data for further model decisions.

## Limitations and responsible use

- The emotion labels are subjective and can overlap. Annotators may disagree about the same text.
- The dataset is English Reddit text, so results may not transfer to other platforms, languages, or communities.
- Emotion labels are imbalanced. Metrics for labels with few examples can vary substantially.
- The app's scores are model outputs, not calibrated confidence or facts about a person's feelings.
- Do not use the model for mental health assessment, employment decisions, or other high-stakes judgments.
- The GoEmotions dataset documentation notes possible dataset biases and potentially problematic content. Review the [dataset documentation](https://github.com/google-research/google-research/tree/master/goemotions) before reuse.

## Data and model attribution

GoEmotions is a Google Research dataset of Reddit comments. The Google Research repository states its datasets are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The pretrained [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) model is listed under Apache-2.0. Follow the applicable licenses and attribution requirements when redistributing data or model artifacts.

Dataset paper: [GoEmotions: A Dataset of Fine-Grained Emotions](https://aclanthology.org/2020.acl-main.372/).