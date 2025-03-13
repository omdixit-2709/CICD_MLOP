#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training script for sentiment analysis.
Trains a Logistic Regression model and performs hyperparameter tuning.
"""

import os
import joblib
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

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
    
    # Save model
    model_path = os.path.join(output_dir, 'sentiment_model.joblib')
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.joblib')
    joblib.dump(metrics, metrics_path)
    
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Load processed data
    logging.info("Loading processed data...")
    X_train, y_train = joblib.load('data/processed/train_data.joblib')
    X_test, y_test = joblib.load('data/processed/test_data.joblib')
    
    # Train model
    model = train_model(X_train, y_train, do_grid_search=True)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and metrics
    save_model(model, metrics) 