import re
import pandas as pd


df = pd.read_csv('../data/Tweets.csv')
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
print(df.head())
