import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from utils import evaluate_performance

# 1. Load the dataset
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# 2. Select Apple Silicon GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 3. Load model into pipeline using MPS backend
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    top_k=None,  # Replaces deprecated return_all_scores=True
    device=device
)

# 4. Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
    # Extract negative (index 0) and positive (index 2) scores from Twitter-RoBERTa
    scores = {item["label"]: item["score"] for item in output}

    negative_score = scores["negative"]
    positive_score = scores["positive"]

    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

# 5. Evaluate performance

evaluate_performance(data["test"]["label"], y_pred)