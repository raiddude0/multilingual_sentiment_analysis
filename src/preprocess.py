import re
import pandas as pd
from transformers import AutoTokenizer
from src.config import MODEL_CHECKPOINT, MAX_LENGTH
from sklearn.model_selection import train_test_split
from datasets import Dataset



df = pd.read_csv('data/Tweets.csv')
df = df[["text", "airline_sentiment"]].rename(columns={"airline_sentiment": "label"})


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text) #url
    text = re.sub(r"@\w+", "", text) #mentions
    text = re.sub(r"[^\x00-\x7F]+", "", text) #non-ascii
    text = re.sub(r"#", "", text) #hashtags
    text = re.sub(r"\s+", " ", text).strip() #whitespace
    return text

df["clean_text"] = df["text"].apply(clean_text)

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

df["label_id"] = df["label"].map(label2id)
df = df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "label"})



tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))    

train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)


train_dataset.save_to_disk("data/train_dataset")
val_dataset.save_to_disk("data/val_dataset")
test_dataset.save_to_disk("data/test_dataset")
