"""Prepare Cardiff NLP's multilingual tweet-sentiment dataset for fine-tuning."""

import argparse
import re
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .config import DATASET_NAME, LANGUAGES, MAX_LENGTH, MODEL_CHECKPOINT, MULTILINGUAL_DATA_DIR


def clean_text(text: str) -> str:
    """Remove tweet metadata while preserving Unicode text, emojis, and accents."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def load_multilingual_splits(dataset_name: str = DATASET_NAME) -> DatasetDict:
    """Load official splits and retain each row's language for later evaluation.

    The dataset's ``all`` configuration merges languages but does not expose their
    provenance. Loading each official configuration retains that information.
    """
    grouped = {"train": [], "validation": [], "test": []}
    for language in LANGUAGES:
        dataset = load_dataset(dataset_name, language)
        for split in grouped:
            grouped[split].append(dataset[split].add_column("language", [language] * len(dataset[split])))
    return DatasetDict({split: concatenate_datasets(parts).shuffle(seed=42) for split, parts in grouped.items()})


def build_datasets(output_dir: Path, checkpoint: str = MODEL_CHECKPOINT) -> None:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_multilingual_splits()
    for split, dataset in datasets.items():
        cleaned = dataset.map(lambda row: {"text": clean_text(row["text"])})
        cleaned.filter(lambda row: bool(row["text"])).map(tokenize, batched=True).save_to_disk(
            str(output_dir / f"{split.replace('validation', 'val')}_dataset")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare multilingual tweet-sentiment data.")
    parser.add_argument("--output-dir", type=Path, default=MULTILINGUAL_DATA_DIR)
    parser.add_argument("--checkpoint", default=MODEL_CHECKPOINT)
    args = parser.parse_args()
    build_datasets(args.output_dir, args.checkpoint)


if __name__ == "__main__":
    main()
