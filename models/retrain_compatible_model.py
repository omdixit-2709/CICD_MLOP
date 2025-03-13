#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retraining script for sentiment analysis model with Docker-compatible library versions.
This script ensures the model is trained with scikit-learn 1.3.0 and numpy 1.24.3.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Print versions to confirm we're using the correct ones
import sklearn
import numpy
print(f"scikit-learn version: {sklearn.__version__}")
print(f"numpy version: {numpy.__version__}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    # Join tokens back to string
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

def load_and_preprocess_data(input_file, sample_size=None):
    """
    Load the Sentiment140 dataset and preprocess it.
    
    Args:
        input_file (str): Path to the input CSV file
        sample_size (int, optional): Number of samples to use (for testing)
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, vectorizer
    """
    logging.info(f"Loading data from {input_file}")
    
    # Load data with proper column names
    # Sentiment140 has 6 columns: polarity, id, date, query, user, text
    column_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(input_file, encoding='latin-1', header=None, names=column_names)
    
    # Apply sampling if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Map sentiment values: 0 for negative (original value is 0), 1 for positive (original value is 4)
    # In Sentiment140: 0 = negative, 2 = neutral, 4 = positive
    # Convert to binary classification by mapping 0 -> 0 (negative) and 4 -> 1 (positive)
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
    
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Class distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # Clean text
    logging.info("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Split data
    X = df['cleaned_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorize text using TF-IDF
    logging.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save processed data with Docker-compatible suffix
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump((X_train_vec, y_train), 'data/processed/train_data_compatible.joblib')
    joblib.dump((X_test_vec, y_test), 'data/processed/test_data_compatible.joblib')
    joblib.dump(vectorizer, 'data/processed/vectorizer_compatible.joblib')
    
    logging.info(f"Processed data saved to data/processed/ with 'compatible' suffix")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

def train_model(X_train, y_train, do_grid_search=True):
    """
    Train a logistic regression model for sentiment analysis.
    
    Args:
        X_train: Training features
        y_train: Training labels
        do_grid_search (bool): Whether to perform grid search for hyperparameter tuning
        
    Returns:
        model: Trained model
    """
    logging.info("Training sentiment analysis model...")
    
    if do_grid_search:
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
        
        # Initialize base model
        base_model = LogisticRegression(random_state=42)
        
        # Perform grid search
        logging.info("Performing hyperparameter tuning with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        logging.info(f"Best parameters: {grid_search.best_params_}")
    else:
        # Train with default parameters
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logging.info("Evaluating model on test data...")
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    
    # Detailed classification report
    report = classification_report(y_test, y_pred)
    logging.info(f"Classification Report:\n{report}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report
    }

def save_model(model, metrics, output_dir='models/saved_models'):
    """
    Save the trained model and its metrics.
    
    Args:
        model: Trained model
        metrics (dict): Evaluation metrics
        output_dir (str): Directory to save the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model with Docker-compatible suffix
    model_path = os.path.join(output_dir, 'sentiment_model_compatible.joblib')
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics_compatible.joblib')
    joblib.dump(metrics, metrics_path)
    
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # First ensure we're using the correct versions
    if sklearn.__version__ != '1.3.0' or not numpy.__version__.startswith('1.24'):
        logging.error(f"Wrong library versions detected! Please install scikit-learn 1.3.0 and numpy 1.24.3")
        logging.error(f"Current versions: scikit-learn {sklearn.__version__}, numpy {numpy.__version__}")
        logging.error("Run: pip install scikit-learn==1.3.0 numpy==1.24.3")
        exit(1)
    
    # Process the data
    input_file = 'data/raw/sentiment140.csv'
    # Use a smaller sample size for faster processing
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = load_and_preprocess_data(input_file, sample_size=100000)
    
    # Train model
    model = train_model(X_train_vec, y_train, do_grid_search=True)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_vec, y_test)
    
    # Save model and metrics
    save_model(model, metrics)
    
    logging.info("Docker-compatible model training complete!")
    logging.info("This model should work properly in the Docker container.") 