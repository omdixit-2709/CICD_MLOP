#!/bin/bash
echo "Setting up a clean environment for Docker-compatible model training..."

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install Docker-compatible versions of dependencies
echo "Installing compatible versions of scikit-learn and numpy..."
pip install scikit-learn==1.3.0 numpy==1.24.3 joblib==1.3.1 pandas==2.0.3
pip install nltk==3.8.1 matplotlib==3.7.2 seaborn==0.12.2

# Run the retraining script
echo "Starting model retraining with compatible versions..."
python models/retrain_compatible_model.py

# After training, update the API file to use the compatible model
echo "Updating API file to use compatible model files..."
cp direct_api.py direct_api_compatible.py
python -c "
with open('direct_api_compatible.py', 'r') as f:
    content = f.read()
content = content.replace('MODEL_PATH = \"models/saved_models/sentiment_model.joblib\"', 
                         'MODEL_PATH = \"models/saved_models/sentiment_model_compatible.joblib\"')
content = content.replace('VECTORIZER_PATH = \"data/processed/vectorizer.joblib\"', 
                         'VECTORIZER_PATH = \"data/processed/vectorizer_compatible.joblib\"')
with open('direct_api_compatible.py', 'w') as f:
    f.write(content)
"

echo "Retraining complete! Now you can build a Docker container with:"
echo "docker build -t sentiment-analysis-api-compatible -f Dockerfile.compatible ."

# Deactivate the virtual environment
deactivate 