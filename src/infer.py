import os
from transformers import pipeline
from config import ID2LABEL, LABEL2ID


# Try local model first, fallback to base model if not available
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "sentiment_model", "multilingual-sentiment-model")

if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    # Fallback for Streamlit Cloud (no local files)
    MODEL_PATH = "xlm-roberta-base"

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=0 if os.environ.get("USE_GPU", "false").lower() == "true" else -1  #0 for GPU -1 for CPU
)


samples = [
    "The flight was delayed three hours and no one told us why.",
    "Merci beaucoup, un vol vraiment agréable !",
    "الخدمة كانت ممتازة والطاقم لطيف جدا",
]

def predict(text):
    """
    Predict sentiment for a single text.
    Returns dict with label name and confidence.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")
    
    result = classifier(text)[0]
    
    return {
        "label": result['label'],
        "confidence": result['score']
    }

def predict_batch(texts):

    """
    Predict sentiment for multiple texts efficiently.
    """
    if not texts or not isinstance(texts, list):
        raise ValueError("Input must be a non-empty list of strings")
    
    results = classifier(texts)
    predictions = []
    for result in results:
        predictions.append({
            "label": result['label'],
            "confidence": result['score']
        })
    return predictions

if __name__ == "__main__":
    print("===single Inference===\n")
    for text in samples:
        try:
            result = predict(text)
            print(f"{text}\n  -> {result['label']} (confidence: {result['confidence']:.2%})\n")
        except Exception as e:
            print(f"Error processing '{text}': {e}\n")
    
    print("\n===batch Inference===\n")
    try:
        batch_results = predict_batch(samples)
        for text, result in zip(samples, batch_results):
            print(f"{text}\n  -> {result['label']} (confidence: {result['confidence']:.2%})\n")
    except Exception as e:
        print(f"Error in batch processing: {e}")