from transformers import pipeline
from textblob import TextBlob
import gradio as gr

# Load Hugging Face toxicity model
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

# TextBlob sentiment analysis
def analyze_sentiment(message):
    polarity = TextBlob(message).sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment

# Detect toxicity
def detect_toxicity(message):
    result = toxicity_model(message)[0]
    label = result["label"]
    score = result["score"]
    return f"{label} ({score:.2f})"

# Combined chat analysis function
def chat_analyze(user, message):
    sentiment = analyze_sentiment(message)
    toxicity = detect_toxicity(message)
    return f"👤 {user}:\n💬 {message}\n🧠 Sentiment: {sentiment}\n☣️ Toxicity: {toxicity}"

# Gradio UI
demo = gr.Interface(
    fn=chat_analyze,
    inputs=[
        gr.Textbox(label="User Name", placeholder="e.g. Sajeetha"),
        gr.Textbox(label="Message", placeholder="Type a message here...")
    ],
    outputs="text",
    title="🧠 Chat Analyzer",
    description="Analyzes Sentiment & Toxicity of messages using TextBlob + Hugging Face BERT",
    theme="monochrome"
)

if __name__ == "__main__":
    demo.launch()
