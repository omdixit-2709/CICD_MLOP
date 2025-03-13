
import os
import time
import joblib
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler(
    'api_logs.log', maxBytes=10485760, backupCount=5, encoding='utf-8'
)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

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

# Mock user database - in production, use a proper database
users = {
    "admin": "password123",
    "user": "user123"
}

def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate user with HTTP Basic Auth"""
    is_correct_username = secrets.compare_digest(credentials.username, "admin") or \
                         secrets.compare_digest(credentials.username, "user")
    is_correct_password = secrets.compare_digest(credentials.password, users.get(credentials.username, ""))
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Model paths
MODEL_PATH = "models/saved_models/sentiment_model_compatible.joblib"

# Load model
model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

# Pydantic models
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str
    probability: float
    processing_time: float

# API endpoints
@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "Sentiment Analysis API is active"}

@app.get("/test")
def test():
    """Test endpoint"""
    return {"message": "Test endpoint is working", "timestamp": time.time()}

@app.get("/health", dependencies=[Depends(authenticate_user)])
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput, dependencies=[Depends(authenticate_user)])
def predict(input_data: TextInput):
    """Predict sentiment from text"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Make prediction
        text = input_data.text
        prediction_proba = model.predict_proba([text])[0]
        sentiment_idx = prediction_proba.argmax()
        sentiment = "positive" if sentiment_idx == 1 else "negative"
        probability = float(prediction_proba[sentiment_idx])
        
        processing_time = time.time() - start_time
        
        return {
            "sentiment": sentiment,
            "probability": probability,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("direct_api_compatible:app", host="0.0.0.0", port=8004, reload=False)
