#gpu unavailable
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda : False

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_CHECKPOINT, LABEL2ID, ID2LABEL
from datasets import load_from_disk
import random
import numpy as np

from transformers import TrainingArguments
from transformers import DataCollatorWithPadding, Trainer
from sklearn.metrics import accuracy_score, f1_score



#loading the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    device_map="cpu"
    
)
#fine-tuning the model
train_dataset = load_from_disk("data/train_dataset")
val_dataset = load_from_disk("data/val_dataset")

#seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

#hyperparameters for training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01, #L2 regularization
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
   
)
#training loop
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    
)


trainer.train()