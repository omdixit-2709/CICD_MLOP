#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing script for the Sentiment140 dataset.
This script handles loading, cleaning, and preprocessing text data.
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

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
    text = text.lower()
    
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
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump((X_train_vec, y_train), 'data/processed/train_data.joblib')
    joblib.dump((X_test_vec, y_test), 'data/processed/test_data.joblib')
    joblib.dump(vectorizer, 'data/processed/vectorizer.joblib')
    
    logging.info(f"Processed data saved to data/processed/")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

if __name__ == "__main__":
    # Process the data
    input_file = 'data/raw/sentiment140.csv'
    # Use a smaller sample size for faster processing during development
    # For the final model, use the full dataset or a larger sample
    load_and_preprocess_data(input_file, sample_size=100000)  # Adjust sample size as needed 