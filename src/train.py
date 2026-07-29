"""Fine-tune the sentiment model."""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import ID2LABEL, LABEL2ID, MODEL_CHECKPOINT, MODEL_DIR, MULTILINGUAL_DATA_DIR, RESULTS_DIR


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def train(data_dir: Path, output_dir: Path, model_dir: Path, checkpoint: str, epochs: float, use_cpu: bool) -> None:
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    )
    args = TrainingArguments(
        output_dir=str(output_dir), eval_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=epochs, weight_decay=0.01, load_best_model_at_end=True,
        metric_for_best_model="f1", logging_steps=50, report_to="none", use_cpu=use_cpu,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=load_from_disk(str(data_dir / "train_dataset")),
        eval_dataset=load_from_disk(str(data_dir / "val_dataset")),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer), processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R sentiment classification.")
    parser.add_argument("--data-dir", type=Path, default=MULTILINGUAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--checkpoint", default=MODEL_CHECKPOINT)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.model_dir, args.checkpoint, args.epochs, args.cpu)


if __name__ == "__main__":
    main()
