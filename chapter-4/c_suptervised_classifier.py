from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

from utils import evaluate_performance

# Load dataset
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

print(f"shape of embeddings for train set: {train_embeddings.shape}")


# Train a logistic regression classifier
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

# Evaluate the classifier
y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)
