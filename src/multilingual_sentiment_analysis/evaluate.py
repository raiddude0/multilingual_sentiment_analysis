"""Evaluate a saved sentiment model and produce plots."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .config import ID2LABEL, MODEL_DIR, MULTILINGUAL_DATA_DIR, RESULTS_DIR


def evaluate(model_dir: Path, test_dataset: Path, output_dir: Path, show: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(output_dir), use_cpu=True, report_to="none"),
        processing_class=tokenizer,
    )
    dataset = load_from_disk(str(test_dataset))
    if "language" not in dataset.column_names:
        raise ValueError("The test dataset must include a language column for per-language evaluation.")
    predictions = trainer.predict(dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    labels = list(sorted(ID2LABEL))
    names = [ID2LABEL[label] for label in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    per_language = {}
    languages = np.array(dataset["language"])
    for language in sorted(set(languages)):
        mask = languages == language
        per_language[language] = classification_report(
            y_true[mask], y_pred[mask], labels=labels, target_names=names, zero_division=0, output_dict=True
        )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"overall": report, "per_language": per_language}, file, ensure_ascii=False, indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)

    state_path = model_dir / "trainer_state.json"
    if state_path.exists():
        with state_path.open(encoding="utf-8") as file:
            history = json.load(file).get("log_history", [])
        train = [(entry["step"], entry["loss"]) for entry in history if "loss" in entry]
        validation = [(entry["step"], entry["eval_loss"]) for entry in history if "eval_loss" in entry]
        if train or validation:
            plt.figure(figsize=(7, 4))
            if train:
                plt.plot(*zip(*train), label="Training loss")
            if validation:
                plt.plot(*zip(*validation), label="Validation loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training vs. Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "loss_curve.png", dpi=150)
    if show:
        plt.show()
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved sentiment model.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--test-dataset", type=Path, default=MULTILINGUAL_DATA_DIR / "test_dataset")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "eval")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    evaluate(args.model_dir, args.test_dataset, args.output_dir, args.show)


if __name__ == "__main__":
    main()
