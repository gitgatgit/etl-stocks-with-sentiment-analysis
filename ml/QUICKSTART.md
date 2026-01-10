# ML Volatility Prediction - Quick Start Guide

Get started with volatility prediction in 5 minutes!

## Prerequisites

1. **Running ETL Pipeline**: Make sure you have stock data collected
   ```bash
   python debug_db.py  # Check if you have data
   ```

2. **Install ML Dependencies**:
   ```bash
   pip install -r ml/requirements.txt
   ```

## Step-by-Step Tutorial

### Step 1: Verify Data Availability

Check that you have enough data for training:

```bash
python debug_db.py
```

You should see at least 100+ stock price records. If not, run the ETL pipeline first:

```bash
python cli.py run
python cli.py trigger
```

Wait for the DAG to complete (check at http://localhost:8080).

### Step 2: Train Your First Model

Train an XGBoost model (takes ~5-10 seconds):

```bash
python cli.py ml-train --ml-model xgboost
```

**What happens:**
- Loads data from PostgreSQL
- Engineers 40+ features (RSI, ATR, Bollinger Bands, etc.)
- Trains XGBoost classifier
- Evaluates on test set
- Saves model to `ml/models/`

**Output:**
```
Training samples: 500
Features: 42
Class distribution:
  low       250
  medium    200
  high       50

✓ Training completed

Test Set Evaluation:
Accuracy: 0.7500
F1 Score (Macro): 0.7200
```

### Step 3: Make Predictions

Predict next-day volatility for all stocks:

```bash
python cli.py ml-predict --save-db
```

**What happens:**
- Loads latest trained model
- Prepares features for most recent data
- Predicts volatility class (low/medium/high)
- Calculates confidence scores
- Saves to database

**Output:**
```
PREDICTIONS
ticker  date        predicted_volatility_class  confidence  prob_low  prob_medium  prob_high
AAPL    2026-01-11  medium                     0.65        0.15      0.65         0.20
MSFT    2026-01-11  low                        0.80        0.80      0.15         0.05
GOOGL   2026-01-11  medium                     0.55        0.25      0.55         0.20
TSLA    2026-01-11  high                       0.70        0.10      0.20         0.70
NVDA    2026-01-11  medium                     0.60        0.20      0.60         0.20
```

### Step 4: Query Predictions

View predictions in database:

```bash
docker-compose exec postgres psql -U airflow -d airflow -c "
SELECT ticker, date, predicted_volatility_class, confidence
FROM analytics.ml_volatility_predictions
ORDER BY date DESC, ticker
LIMIT 10;
"
```

Or export to CSV:

```bash
python cli.py ml-predict --output my_predictions.csv
```

## Advanced Usage

### Train on Specific Date Range

Focus on recent data:

```bash
python cli.py ml-train --min-date 2025-01-01 --max-date 2025-12-31
```

### Predict Specific Tickers

Only predict for selected stocks:

```bash
python cli.py ml-predict --tickers AAPL TSLA --save-db
```

### Try Random Forest Instead

```bash
python cli.py ml-train --ml-model random_forest
```

### Use Direct Python API

For more control, use the Python modules directly:

```python
from ml.data_loader import StockDataLoader
from ml.feature_engineering import engineer_features
from ml.train import VolatilityModelTrainer

# Load data
loader = StockDataLoader()
df = loader.load_training_data()

# Engineer features
df_features = engineer_features(df, for_prediction=False)

# Train model
trainer = VolatilityModelTrainer(model_type='xgboost')
# ... (see train.py for full example)
```

## Automated Training with Airflow

### Option 1: Manual Trigger

1. Go to Airflow UI: http://localhost:8080
2. Find `ml_volatility_pipeline` DAG
3. Click "Trigger DAG"

### Option 2: Scheduled Retraining

Edit `airflow/dags/ml_volatility_pipeline.py`:

```python
dag = DAG(
    'ml_volatility_pipeline',
    schedule_interval='@weekly',  # Train every week
    ...
)
```

## Understanding the Results

### Volatility Classes

- **Low (< 2%)**: Stable price movement, typical of blue-chip stocks
- **Medium (2-5%)**: Moderate volatility, common during normal trading
- **High (> 5%)**: Significant price swings, often news-driven

### Confidence Scores

- **> 0.8**: Very confident prediction
- **0.6 - 0.8**: Moderately confident
- **< 0.6**: Less confident, borderline case

### Model Performance Metrics

- **Accuracy**: Overall correct predictions (aim for > 70%)
- **F1 Score**: Balanced precision/recall (aim for > 0.65)
- **Confusion Matrix**: Shows which classes get confused

## Troubleshooting

### "Module not found: ml"

Make sure you're running from the project root:

```bash
cd /path/to/etl-stocks-with-sentiment-analysis
python cli.py ml-train
```

### "Not enough data for training"

Run the ETL pipeline to collect more data:

```bash
python cli.py trigger
```

Wait 24 hours and collect daily data for better models.

### "XGBoost not available"

Install XGBoost:

```bash
pip install xgboost>=2.0.0
```

Or use Random Forest (no XGBoost required):

```bash
python cli.py ml-train --ml-model random_forest
```

### Predictions seem random

Common causes:
- Not enough training data (< 100 samples)
- High class imbalance (check class distribution)
- Data quality issues

Solution: Collect more data over time.

## Next Steps

1. **Integrate with Metabase**: Create dashboards visualizing predictions
2. **Backtest**: Compare predictions against actual next-day volatility
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Add Features**: Incorporate external data (VIX, economic indicators)
5. **Multi-step Predictions**: Predict 3-day, 5-day volatility
6. **LSTM Models**: Try deep learning for time-series patterns

## Resources

- **Full Documentation**: See [ml/README.md](README.md)
- **Feature Engineering**: See `ml/feature_engineering.py`
- **Model Code**: See `ml/train.py` and `ml/predict.py`
- **Main Project**: See [../README.md](../README.md)

Happy predicting! 🚀
