#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics collection module for model monitoring.
"""

import time
from prometheus_client import Counter, Histogram, start_http_server, Gauge

# Define metrics
PREDICTION_COUNT = Counter(
    'sentiment_prediction_count', 
    'Number of sentiment predictions made',
    ['sentiment', 'status']
)

PREDICTION_LATENCY = Histogram(
    'sentiment_prediction_latency_seconds', 
    'Time taken for sentiment prediction',
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5]
)

MODEL_CONFIDENCE = Histogram(
    'sentiment_model_confidence', 
    'Confidence of sentiment predictions',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

API_REQUEST_COUNT = Counter(
    'api_request_count', 
    'Number of API requests',
    ['endpoint', 'method', 'status_code']
)

API_REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds', 
    'API request latency in seconds',
    ['endpoint', 'method'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
)

MODEL_LOADED = Gauge(
    'model_loaded', 
    'Whether the model is loaded (1) or not (0)'
)

def start_metrics_server(port=8001):
    """
    Start the Prometheus metrics server.
    
    Args:
        port (int): Port to run the metrics server on
    """
    start_http_server(port)
    print(f"Metrics server started on port {port}")

def record_prediction(sentiment, confidence, latency, status="success"):
    """
    Record a prediction metric.
    
    Args:
        sentiment (str): Predicted sentiment ("positive" or "negative")
        confidence (float): Prediction confidence
        latency (float): Prediction latency in seconds
        status (str): Prediction status ("success" or "error")
    """
    PREDICTION_COUNT.labels(sentiment=sentiment, status=status).inc()
    PREDICTION_LATENCY.observe(latency)
    MODEL_CONFIDENCE.observe(confidence)

def record_api_request(endpoint, method, status_code, latency):
    """
    Record an API request metric.
    
    Args:
        endpoint (str): API endpoint path
        method (str): HTTP method
        status_code (int): HTTP status code
        latency (float): Request latency in seconds
    """
    API_REQUEST_COUNT.labels(
        endpoint=endpoint, 
        method=method, 
        status_code=status_code
    ).inc()
    
    API_REQUEST_LATENCY.labels(
        endpoint=endpoint,
        method=method
    ).observe(latency)

def set_model_loaded(is_loaded):
    """
    Set the model loaded gauge.
    
    Args:
        is_loaded (bool): Whether the model is loaded
    """
    MODEL_LOADED.set(1 if is_loaded else 0)

class MetricsMiddleware:
    """
    FastAPI middleware for recording request metrics.
    """
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        latency = time.time() - start_time
        
        record_api_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency=latency
        )
        
        return response 