# Models

This directory contains all the model-related code for the sentiment analysis project.

## Structure

- `training.py`: Script for training the sentiment analysis model
- `evaluation.py`: Script for evaluating model performance
- `saved_models/`: Directory where trained models are saved

## Training Process

The model training process includes the following steps:

1. Loading preprocessed data
2. Training a Logistic Regression model
3. Performing hyperparameter tuning using GridSearchCV
4. Evaluating the model on the test set
5. Saving the best model for inference

## Model Selection

We use Logistic Regression as our baseline model because:
- It's computationally efficient
- It performs well for text classification tasks
- It provides interpretable results
- It works well with high-dimensional sparse data (like TF-IDF vectors)

## Hyperparameter Tuning

The hyperparameter grid search explores:
- Regularization strength (`C`)
- Penalty type (`l1` or `l2`)
- Solver algorithm

## Training a New Model

To train a new model, run:

```bash
python models/training.py
```

This will:
1. Load the processed data from `data/processed/`
2. Train the model
3. Save the trained model to `models/saved_models/`

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Model Artifacts

After training, the following artifacts are saved:
- `sentiment_model.joblib`: The trained model
- `model_metrics.joblib`: Model performance metrics 