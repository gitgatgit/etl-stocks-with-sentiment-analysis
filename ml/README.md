# Volatility Prediction ML Pipeline

Machine learning pipeline for predicting stock volatility classification using historical price data and sentiment analysis.

## Overview

This module predicts **next-day volatility** for stocks by classifying them into three categories:
- **Low volatility**: < 2% intraday range
- **Medium volatility**: 2% - 5% intraday range
- **High volatility**: > 5% intraday range

Volatility is measured as: `(high - low) / close * 100`

## Features Used

### Technical Indicators
- **RSI** (Relative Strength Index)
- **ATR** (Average True Range)
- **Bollinger Bands Width**
- Price returns (1-day, 3-day, 5-day)

### Historical Statistics
- Rolling volatility (3, 5, 10, 20-day windows)
- Rolling return statistics
- Volume ratios

### Sentiment Features
- Grok-generated sentiment (positive/negative/neutral)
- Topic classification (earnings/macro/company-specific/speculation)

### Temporal Features
- Day of week
- Month

## Installation

Install ML dependencies:

```bash
pip install -r ml/requirements.txt
```

## Usage

### 1. Train a Model

Train using XGBoost (recommended):

```bash
python -m ml.train --model xgboost
```

Train using Random Forest:

```bash
python -m ml.train --model random_forest
```

**Options:**
- `--model`: Model type (`xgboost` or `random_forest`)
- `--min-date`: Minimum date for training data (YYYY-MM-DD)
- `--max-date`: Maximum date for training data (YYYY-MM-DD)
- `--test-size`: Proportion for test set (default: 0.2)
- `--val-size`: Proportion for validation set (default: 0.1)
- `--output-dir`: Directory to save model (default: `ml/models`)

**Example with date filtering:**
```bash
python -m ml.train --model xgboost --min-date 2024-01-01 --max-date 2024-12-31
```

### 2. Make Predictions

Predict next-day volatility using the latest trained model:

```bash
python -m ml.predict
```

**Options:**
- `--model`: Path to model file (default: `ml/models/latest.pkl`)
- `--metadata`: Path to metadata file (default: `ml/models/latest_metadata.json`)
- `--tickers`: Specific tickers to predict (default: all)
- `--save-db`: Save predictions to database
- `--output`: Save predictions to CSV file

**Examples:**

Predict for specific tickers and save to database:
```bash
python -m ml.predict --tickers AAPL MSFT --save-db
```

Predict and save to CSV:
```bash
python -m ml.predict --output predictions.csv
```

Use a specific model:
```bash
python -m ml.predict --model ml/models/volatility_xgboost_20260110_120000.pkl
```

## Model Architecture

### XGBoost Classifier (Recommended)
- **n_estimators**: 200
- **max_depth**: 6
- **learning_rate**: 0.1
- **objective**: multi:softmax (3 classes)

### Random Forest Classifier
- **n_estimators**: 200
- **max_depth**: 10
- **min_samples_split**: 10
- **min_samples_leaf**: 4

## Data Pipeline

1. **Data Loading**: Fetch from `analytics.fct_prices_with_grok` table
2. **Feature Engineering**: Calculate 40+ features including technical indicators
3. **Train/Val/Test Split**: Time-series aware splitting (chronological)
4. **Model Training**: Train with validation set for early stopping
5. **Evaluation**: Classification metrics (accuracy, F1, confusion matrix)
6. **Prediction**: Forecast next-day volatility with confidence scores

## Database Schema

Predictions are stored in `analytics.ml_volatility_predictions`:

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| ticker | VARCHAR(10) | Stock symbol |
| date | DATE | Prediction date |
| predicted_volatility_class | VARCHAR(10) | low/medium/high |
| predicted_volatility | NUMERIC | Predicted volatility value |
| confidence | NUMERIC | Model confidence (0-1) |
| model_version | VARCHAR(50) | Model identifier |
| created_at | TIMESTAMP | Prediction timestamp |

## Module Structure

```
ml/
├── __init__.py              # Module initialization
├── data_loader.py           # Database interaction utilities
├── feature_engineering.py   # Feature calculation and preparation
├── train.py                 # Model training script
├── predict.py               # Prediction script
├── requirements.txt         # ML dependencies
├── models/                  # Trained models directory
│   ├── latest.pkl           # Symlink to latest model
│   └── latest_metadata.json # Symlink to latest metadata
└── README.md                # This file
```

## Example Workflow

```bash
# 1. Ensure data is available
python debug_db.py

# 2. Install ML dependencies
pip install -r ml/requirements.txt

# 3. Train a model
python -m ml.train --model xgboost

# 4. Make predictions
python -m ml.predict --save-db

# 5. Query predictions
psql -h localhost -U airflow -d airflow -c \
  "SELECT * FROM analytics.ml_volatility_predictions ORDER BY date DESC LIMIT 10;"
```

## Performance Notes

- **Training time**: ~1-5 seconds per 1000 samples
- **Minimum data**: Recommended 100+ samples for meaningful training
- **Feature count**: ~40-50 features depending on available data
- **Memory usage**: ~100-200 MB for typical datasets

## Future Enhancements

Potential improvements:
- LSTM/GRU models for better time-series modeling
- Hyperparameter tuning with GridSearch/Optuna
- Multi-step ahead predictions
- Volatility value regression (in addition to classification)
- Real-time prediction API endpoint
- Model retraining automation in Airflow
- A/B testing framework for model comparison
