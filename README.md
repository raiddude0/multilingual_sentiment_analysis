# Multilingual Sentiment Analysis

An end-to-end NLP project for three-class sentiment classification: **negative**, **neutral**, and **positive**. The repository contains two XLM-RoBERTa experiments:

1. an archived English airline-tweet model, and
2. the current general multilingual Twitter-sentiment model.

The current model is exposed through a Gradio app and is evaluated both overall and separately for every supported language.

## Results at a glance

| Experiment | Data | Evaluation | Result |
| --- | --- | --- | --- |
| Airline tweets (legacy) | English airline tweets | Validation set | 84.24% accuracy, 0.842 weighted F1 |
| General multilingual (current) | 8-language Twitter sentiment dataset | Held-out test set (6,960 tweets) | 66.52% accuracy, 0.663 macro F1 |

The two scores are **not directly comparable**: the airline model was evaluated on a narrow, English-only domain, whereas the multilingual model is evaluated across eight languages and broader Twitter content.

### Current multilingual model: test macro F1 by language

| Language | Macro F1 |
| --- | ---: |
| German | 0.731 |
| French | 0.721 |
| Portuguese | 0.699 |
| English | 0.692 |
| Spanish | 0.667 |
| Arabic | 0.647 |
| Italian | 0.625 |
| Hindi | 0.516 |

The neutral class is the most difficult overall (0.588 F1). Hindi is the weakest supported language, so this project does not claim equal quality across all languages.

## How it works

1. `src.preprocess` downloads and prepares the multilingual dataset.
2. Text is cleaned by removing URLs and mentions, normalizing whitespace, and removing only the `#` marker. Unicode text, accents, emojis, and non-Latin scripts are preserved.
3. `src.train` fine-tunes `xlm-roberta-base` for three-class sequence classification.
4. `src.evaluate` reports overall metrics, per-language metrics, and a confusion matrix.
5. `app.py` provides single-text and batch inference with the saved model.

## Dataset

The current experiment uses [Cardiff NLP's Tweet Sentiment Multilingual dataset](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual). It provides the same label mapping used by this project:

```text
0 = negative
1 = neutral
2 = positive
```

It includes Arabic, English, French, German, Hindi, Italian, Portuguese, and Spanish. The preparation script loads each language configuration separately so it can preserve the official train/validation/test split and retain the language for evaluation.

## Repository layout

```text
src/
  config.py       # shared paths, labels, and dataset settings
  preprocess.py   # dataset download, cleaning, tokenization, and saving
  train.py        # fine-tuning entry point
  evaluate.py     # test and per-language evaluation
  infer.py        # lazy model loading and inference helpers
tests/            # regression tests for UI and preprocessing behavior
data/
  airline_tweets/ # archived English-only source data and prepared splits
  multilingual/   # prepared multilingual splits
results/
  airline_tweets/ # archived English-only checkpoints
  multilingual/   # current checkpoints and evaluation outputs
sentiment_model/
  airline-tweets-sentiment-model/         # archived English-only model
  general-multilingual-sentiment-model/   # current app model
```

Data, checkpoints, and model weights are intentionally ignored by Git.

## Setup

Python **3.11 (64-bit)** is recommended. For NVIDIA RTX 50-series GPUs, install a CUDA 12.8 PyTorch wheel before installing the remaining dependencies.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Run the current multilingual pipeline

Download, clean, tokenize, and save the official multilingual splits:

```powershell
python -m src.preprocess
```

Fine-tune on GPU (do not pass `--cpu`):

```powershell
python -m src.train
```

Evaluate the held-out test set:

```powershell
python -m src.evaluate
```

Results are written to `results/multilingual/eval/`, including `metrics.json` and `confusion_matrix.png`.

Launch the web app:

```powershell
python app.py
```

To load a model from another location, set `SENTIMENT_MODEL_PATH` before launching the app.

## Training configuration

- Base checkpoint: `xlm-roberta-base`
- Maximum token length: 128
- Epochs: 3
- Learning rate: 2e-5 with the Trainer's linear schedule
- Train/evaluation batch sizes: 16 / 32
- Best checkpoint criterion: weighted F1 on the validation set
- Random seed: 42

The multilingual training run completed in approximately 4 minutes 40 seconds on an NVIDIA RTX 5060 Laptop GPU.

## Limitations and next steps

- The model was evaluated on Twitter-style text only; it is not validated for reviews, support tickets, or other domains.
- It is evaluated for eight languages, not all languages supported by the XLM-R tokenizer.
- Hindi and neutral-sentiment performance require further improvement.
- A useful next experiment is to initialize from a Twitter-adapted multilingual encoder such as XLM-T and compare per-language macro F1 against this baseline.

## Testing

```powershell
python -m pytest -q
```

## Attribution

The multilingual dataset is provided by Cardiff NLP. See its [dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) and the associated [XLM-T paper](https://aclanthology.org/2022.lrec-1.27/).
