FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in the correct order with specific versions
# First install numpy and scikit-learn explicitly to ensure correct versions
RUN pip install --no-cache-dir numpy==1.26.0 scikit-learn==1.6.1 joblib==1.3.1
# Then install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt
# Verify installed versions
RUN pip list | grep -E "numpy|scikit-learn|joblib"

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/saved_models logs

# Run NLTK downloads
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# Verify model compatibility
RUN python -c "import joblib, os; print('Model path exists:', os.path.exists('models/saved_models/sentiment_model.joblib')); print('Vectorizer path exists:', os.path.exists('data/processed/vectorizer.joblib'))"

# Expose port
EXPOSE 8004

# Command to run the application
CMD ["python", "direct_api.py"] 