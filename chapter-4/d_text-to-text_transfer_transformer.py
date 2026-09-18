import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from utils import evaluate_performance

# Load dataset
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Apply prompt to each example
prompt = (
    "Is the following movie review positive or negative? "
    "Answer with only positive or negative: "
)

data = data.map(
    lambda example: {"t5": prompt + example["text"]}
)

# Select Apple Silicon GPU if available
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load FLAN-T5
model_path = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to(device)

# Run inference
# Run inference in batches
y_pred = []

batch_size = 16

for i in tqdm(
    range(0, len(data["test"]), batch_size),
    desc="Classifying"
):
    batch = data["test"]["t5"][i:i + batch_size]

    # Tokenize the entire batch
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5
        )

    # Decode all predictions
    predictions = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    # Convert text predictions to labels
    for text in predictions:
        text = text.strip().lower()

        if "negative" in text:
            y_pred.append(0)
        else:
            y_pred.append(1)
# Evaluate
evaluate_performance(
    data["test"]["label"],
    y_pred
)