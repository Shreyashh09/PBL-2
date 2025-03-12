
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Path to your saved model
MODEL_PATH = "best_bert_model.pth"

# Class mapping (these should match your original labels)
# Replace with your actual label names from the LabelEncoder
SENTIMENT_CLASSES = [
    "Negative",
    "Neutral",
    "Positive",
]  # Example - update based on your actual classes

# Tokenizer setup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=False, resume_download=True)


# Load model
def load_model():
    logger.info("Loading model...")
    model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=len(SENTIMENT_CLASSES),
    local_files_only=False,
    resume_download=True
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("Custom model weights loaded successfully")
    except FileNotFoundError:
        logger.warning(f"Custom model file {MODEL_PATH} not found. Using base BERT model.")
    except Exception as e:
        logger.error(f"Error loading custom model: {str(e)}")
    
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    return model


# Apply temperature scaling to logits
def apply_temperature(logits, temperature=1.0):
    """Apply temperature scaling to logits."""
    return logits / temperature


# Apply class weights to correct potential bias
def apply_class_weights(logits, class_weights=None):
    """Apply class weights to correct bias."""
    if class_weights is None:
        return logits

    # Convert weights to tensor and move to device
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return logits * weights_tensor


# Prediction function with adjustable parameters
def predict_sentiment(
    tweet_text, model, temperature=1.0, class_weights=None, threshold=None
):
    if not tweet_text.strip():
        return {label: 0.0 for label in SENTIMENT_CLASSES}

    # Tokenize input
    inputs = tokenizer.encode_plus(
        tweet_text,
        None,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # Get logits
        logits = outputs.logits

        # Apply temperature scaling
        if temperature != 1.0:
            logits = apply_temperature(logits, temperature)

        # Apply class weights if provided
        if class_weights is not None:
            logits = apply_class_weights(logits, class_weights)

        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)

    # Get top class and probability
    top_p, top_class = torch.max(probs, dim=1)

    # Create dictionary mapping class names to probabilities
    results = {
        SENTIMENT_CLASSES[i]: float(probs[0][i]) for i in range(len(SENTIMENT_CLASSES))
    }

    # Log prediction details for debugging
    logger.info(f"Tweet: {tweet_text[:50]}...")
    logger.info(f"Raw logits: {logits[0].tolist()}")
    logger.info(f"Probabilities: {probs[0].tolist()}")
    logger.info(
        f"Predicted class: {SENTIMENT_CLASSES[top_class.item()]} ({top_p.item():.4f})"
    )

    return results


# Load model once at startup
model = load_model()

# Default class weights to counter potential bias (increase weight of non-neutral classes)
# These values can be adjusted based on your model's behavior
default_class_weights = [
    1.5,
    1.0,
    1.5,
]  # Example: Increase weight for Negative and Positive


# Define Gradio interface function
def sentiment_analysis(tweet, temperature, use_weights):
    # Use class weights if selected
    weights = default_class_weights if use_weights else None

    # Analyze tweet
    results = predict_sentiment(
        tweet, model, temperature=temperature, class_weights=weights
    )

    return results


# Create Gradio interface
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter a tweet about COVID-19 here..."),
        gr.Slider(
            minimum=0.5,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Temperature",
            info="Lower values make predictions more confident, higher values make them more conservative",
        ),
        gr.Checkbox(
            label="Apply Class Weights",
            value=True,
            info="Apply weights to counter potential bias toward neutral class",
        ),
    ],
    outputs=gr.Label(num_top_classes=len(SENTIMENT_CLASSES)),
    title="COVID-19 Tweet Sentiment Analyzer",
    description="This application analyzes the sentiment of COVID-19 related tweets as Positive, Neutral, or Negative.",
    examples=[
        [
            "The covid death toll exceeds 1000 in the USA. In my locality, 2 young boys died due to the virus. It is spreading exponentially. ",
            0.7,
            True,
        ],
        [
            "The vaccine rollout has been well-organized in my neighborhood! This means good progress",
            0.8,
            True,
        ],
        ["COVID-19 restrictions are being lifted as cases decline.", 1.0, True],
        [
            "Not sure how to feel about the new variant, just waiting for more data.",
            1.0,
            True,
        ],
        ["I hate how COVID has affected everyone's lives so negatively.", 0.8, True],
        ["I'm happy about the progress we've made against COVID-19!", 0.8, True],
    ],
    examples_per_page=6,
    live=False,
    article="""
    ### About
    This sentiment analysis model was trained on COVID-19 related tweets. 
    It classifies tweets into sentiment categories.
    
    ### Adjusting Predictions
    - **Temperature**: Controls the "confidence" of predictions. Lower values (0.5-0.9) make predictions more decisive, 
      higher values (1.1-2.0) make them more conservative.
    - **Class Weights**: Can help counter bias if the model tends to favor the neutral class too often.
    
    ### Debugging
    If most tweets are being classified as "Neutral", try:
    1. Lowering the temperature (0.7-0.8) to make predictions more decisive
    2. Keeping "Apply Class Weights" checked to reduce bias toward the neutral class
    3. Using more strongly-worded test examples
    """,
)

# Launch the app
if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(share=True)  # Set share=False if you don't want a public link

