import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os

print("Creating a Docker-compatible sentiment analysis model...")

# Create directories if they don't exist
os.makedirs('models/saved_models', exist_ok=True)

# Create a simple dataset for demonstration
texts = [
    "I love this product! It's amazing and works perfectly.",
    "This is great, I'm very happy with it.",
    "Excellent service and quality.",
    "I'm satisfied with my purchase.",
    "Works exactly as described, very pleased.",
    "This is terrible. I hate it and it doesn't work at all.",
    "Worst product ever, complete waste of money.",
    "Very disappointed, don't buy this.",
    "Poor quality and bad customer service.",
    "It broke after one use, would not recommend."
]

sentiments = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative

# Create a simple pipeline with CountVectorizer and MultinomialNB
print("Training a simple model...")
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(texts, sentiments)

# Save the model
print("Saving Docker-compatible model...")
joblib.dump(pipeline, 'models/saved_models/docker_model.joblib')

print("Model saved to models/saved_models/docker_model.joblib")

# Test the model
test_texts = [
    "I really like this product",
    "This is awful"
]

predictions = pipeline.predict(test_texts)
probabilities = pipeline.predict_proba(test_texts)

print("\nModel test results:")
for i, text in enumerate(test_texts):
    sentiment = "positive" if predictions[i] == 1 else "negative"
    probability = probabilities[i].max()
    print(f"Text: '{text}'")
    print(f"Prediction: {sentiment}")
    print(f"Confidence: {probability:.4f}\n") 