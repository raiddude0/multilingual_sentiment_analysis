from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AIRLINE_DATA_DIR = DATA_DIR / "airline_tweets"
MULTILINGUAL_DATA_DIR = DATA_DIR / "multilingual"
AIRLINE_RESULTS_DIR = PROJECT_ROOT / "results" / "airline_tweets"
RESULTS_DIR = PROJECT_ROOT / "results" / "multilingual"
AIRLINE_MODEL_DIR = PROJECT_ROOT / "sentiment_model" / "airline-tweets-sentiment-model"
MODEL_DIR = PROJECT_ROOT / "sentiment_model" / "general-multilingual-sentiment-model"

MODEL_CHECKPOINT = "xlm-roberta-base"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
MAX_LENGTH = 128
DATASET_NAME = "cardiffnlp/tweet_sentiment_multilingual"
LANGUAGES = ("arabic", "english", "french", "german", "hindi", "italian", "portuguese", "spanish")
