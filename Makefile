.PHONY: setup data train api test docker deploy clean

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

download-data:
	python -c "import os; os.makedirs('data/raw', exist_ok=True)"
	python -c "import pandas as pd; import os; url='https://raw.githubusercontent.com/kazanova/sentiment140/master/training.1600000.processed.noemoticon.csv'; df = pd.read_csv(url, encoding='latin-1', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text']); df.to_csv('data/raw/sentiment140.csv', index=False)"

process-data:
	python data/data_processing.py

train:
	python models/training.py

evaluate:
	python models/evaluation.py

api:
	uvicorn api.app:app --reload

test:
	pytest tests/

docker-build:
	docker build -t sentiment-analysis-api .

docker-run:
	docker run -p 8000:8000 sentiment-analysis-api

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete 