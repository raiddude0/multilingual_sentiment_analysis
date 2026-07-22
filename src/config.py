MODEL_CHECKPOINT = "xlm-roberta-base"   # swap to "distilbert-base-uncased" for an English-only baseline
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 128