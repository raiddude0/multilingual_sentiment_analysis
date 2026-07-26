import os
import json

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.config import LABEL2ID

import matplotlib.pyplot as plt
import seaborn as sns

model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-1923")
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-1923")
test_ds = load_from_disk("data/test_dataset")

os.environ["ACCELERATE_USE_CPU"] = "true"

training_args = TrainingArguments(
	output_dir="./results/eval",
	use_cpu=True,
	report_to="none",
)

#not many arguments are passed since we only use .predict() 
trainer = Trainer(model=model, args=training_args, processing_class=tokenizer)

predictions = trainer.predict(test_ds)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

print(classification_report(y_true, y_pred, target_names=list(LABEL2ID.keys())))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL2ID.keys(), yticklabels=LABEL2ID.keys())
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


#plot loss
with open("./results/checkpoint-1923/trainer_state.json", "r", encoding="utf-8") as f:
	trainer_state = json.load(f)

log_history = trainer_state["log_history"]
train_steps = [x["step"] for x in log_history if "loss" in x]
train_loss = [x["loss"] for x in log_history if "loss" in x]
eval_steps = [x["step"] for x in log_history if "eval_loss" in x]
eval_loss = [x["eval_loss"] for x in log_history if "eval_loss" in x]

plt.figure(figsize=(7, 4))
plt.plot(train_steps, train_loss, label="Training Loss")
plt.plot(eval_steps, eval_loss, label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()
