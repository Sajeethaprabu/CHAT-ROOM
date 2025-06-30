import gradio as gr
from textblob import TextBlob
import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")  # Your API key

def analyze_sentiment(message):
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    return "Neutral"

def detect_toxicity(message):
    response = requests.post(
        "https://api-inference.huggingface.co/models/unitary/toxic-bert",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": message},
    ) 
    r = response.json()
    label = r[0]["label"]
    score = r[0]["score"]
    return f"{label} ({score:.2f})"

def analyze(user, message):
    return (
        f"👤 {user}\n"
        f"💬 {message}\n"
        f"🧠 Sentiment: {analyze_sentiment(message)}\n"
        f"☣️ Toxicity: {detect_toxicity(message)}"
    )

iface = gr.Interface(
    fn=analyze,
    inputs=[gr.Textbox(label="User"), gr.Textbox(label="Message")],
    outputs="text",
    title="Chat Sentiment + Toxicity Analyzer",
)

if __name__ == "__main__":
    iface.launch()
