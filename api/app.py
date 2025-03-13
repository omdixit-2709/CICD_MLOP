#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAPI application for sentiment analysis model serving.
"""

import os
import time
import joblib
import uvicorn
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
from loguru import logger

# Configure logger
logger.add(
    "logs/api.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time} {level} {message}"
)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis of text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()

# Mock users database - in production, use a proper database and hash passwords
USERS = {
    "admin": "password123",
    "user": "pass456"
}

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Authenticate user with HTTP Basic authentication.
    """
    username = credentials.username
    password = credentials.password
    
    if username not in USERS or not secrets.compare_digest(USERS[username], password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username

# Load model and vectorizer
try:
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logger.info("Loading model and vectorizer...")
    MODEL_PATH = "models/saved_models/sentiment_model.joblib"
    VECTORIZER_PATH = "data/processed/vectorizer.joblib"
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Model and vectorizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or vectorizer: {str(e)}")
    # We'll initialize these as None and handle in endpoints
    model = None
    vectorizer = None

class PredictionRequest(BaseModel):
    """
    Request model for sentiment prediction.
    """
    text: str

class PredictionResponse(BaseModel):
    """
    Response model for sentiment prediction.
    """
    sentiment: str
    probability: float
    processing_time: float

@app.get("/")
def read_root():
    """
    Root endpoint.
    """
    return {"message": "Welcome to the Sentiment Analysis API", "status": "active"}

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or vectorizer not loaded"
        )
    return {
        "status": "healthy",
        "model": "loaded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(
    request: PredictionRequest,
    username: str = Depends(get_current_user)
):
    """
    Predict sentiment from text.
    """
    # Check if model and vectorizer are loaded
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or vectorizer not loaded"
        )
    
    try:
        # Log request
        logger.info(f"Prediction request received from user: {username}")
        
        # Preprocess and vectorize input text
        start_time = time.time()
        text_vectorized = vectorizer.transform([request.text])
        
        # Make prediction
        prediction_prob = model.predict_proba(text_vectorized)[0]
        predicted_class = model.predict(text_vectorized)[0]
        processing_time = time.time() - start_time
        
        # Determine sentiment and probability
        sentiment = "positive" if predicted_class == 1 else "negative"
        probability = prediction_prob[1] if predicted_class == 1 else prediction_prob[0]
        
        # Log result
        logger.info(
            f"Prediction completed: sentiment={sentiment}, "
            f"probability={probability:.4f}, time={processing_time:.4f}s"
        )
        
        # Return response
        return PredictionResponse(
            sentiment=sentiment,
            probability=float(probability),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000) 