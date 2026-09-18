import re
from openai import OpenAI
import torch
from datasets import load_dataset
from tqdm import tqdm

from utils import evaluate_performance

# Load dataset
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Point the OpenAI client to your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def chat_generation(prompt, document, model="llama3.2"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    return chat_completion.choices[0].message.content

def extract_binary_label(response_text):
    """Safely extracts 0 or 1 from LLM output."""
    match = re.search(r'\b(0|1)\b', response_text)
    if match:
        return int(match.group(1))
    return 1 if "positive" in response_text.lower() else 0

prompt = """
Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""

# 1. Limit input text to the first 100 items using [:100]
predictions = [
    chat_generation(prompt, doc) for doc in tqdm(data["test"]["text"][:100])
]

# 2. Extract parsed 0/1 predictions safely
y_pred = [extract_binary_label(pred) for pred in predictions]

# 3. Limit true labels to the first 100 items to match prediction length
evaluate_performance(data["test"]["label"][:100], y_pred)