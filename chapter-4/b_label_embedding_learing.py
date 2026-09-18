import numpy as np

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

from utils import evaluate_performance

# Load dataset
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

label_embeddings = model.encode(["A very negative movie review",  "A very positive movie review"])

test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

evaluate_performance(data["test"]["label"], y_pred)
