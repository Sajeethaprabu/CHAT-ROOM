from textblob import TextBlob
from transformers import pipeline

toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

def analyze_sentiment(message):
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

def detect_toxicity(message):
    result = toxicity_model(message)[0]
    return result["label"], round(result["score"], 2)
