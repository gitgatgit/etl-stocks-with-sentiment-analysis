# MLOps: Volatility Forecasting Models

Machine Learning Operations infrastructure for stock volatility prediction using GARCH and LSTM models.

## Overview

This MLOps system provides end-to-end machine learning lifecycle management for volatility forecasting:

- **Model Training**: GARCH (traditional) and LSTM (deep learning) models
- **Model Registry**: MLflow for versioning and tracking
- **Model Serving**: FastAPI endpoints for real-time predictions
- **Model Monitoring**: Performance metrics and drift detection
- **Automation**: Airflow DAGs for training and inference

## Architecture

```
Stock Data (PostgreSQL)
    ↓
ML Training Pipeline (Airflow DAG)
    ↓
Model Registry (MLflow)
    ↓
Model Serving API (FastAPI)
    ↓
Predictions Storage (PostgreSQL)
    ↓
BI Dashboard (Streamlit)
```

## Components

### 1. Volatility Models

#### GARCH Model (`models/volatility_garch.py`)
- **Type**: Generalized Autoregressive Conditional Heteroskedasticity
- **Purpose**: Traditional financial volatility modeling
- **Features**:
  - GARCH(1,1) specification
  - Log returns calculation
  - Multi-horizon forecasting
  - Model evaluation metrics (MSE, MAE, RMSE)

#### LSTM Model (`models/volatility_lstm.py`)
- **Type**: Long Short-Term Memory neural network
- **Purpose**: Deep learning-based volatility prediction
- **Features**:
  - 2-layer LSTM architecture
  - Dropout regularization
  - Sequence length: 20 days
  - PyTorch implementation
  - GPU support

### 2. MLflow Tracking Server

**Access**: http://localhost:5000

**Features**:
- Experiment tracking
- Model versioning
- Metrics logging
- Artifact storage
- Model registry

**Storage**:
- Backend: PostgreSQL
- Artifacts: Local file system (`/mlflow/artifacts`)

### 3. Model Serving API

**Access**: http://localhost:8000

**Endpoints**:

```
GET  /              - API information
GET  /health        - Health check
GET  /models        - List all models
POST /predict       - Generate predictions
GET  /predictions/{ticker}  - Get stored predictions
```

**Example Usage**:

```bash
# Generate 5-day volatility forecast
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "model_type": "GARCH",
    "horizon": 5
  }'
```

### 4. Airflow DAGs

#### Training DAG: `ml_train_volatility_models`
- **Schedule**: Weekly on Mondays at 2 AM
- **Tasks**:
  1. Train GARCH models for all tickers
  2. Train LSTM models for all tickers
  3. Evaluate and compare models
  4. Store metrics in database

**Trigger manually**:
```bash
# Via Airflow UI or CLI
docker-compose exec airflow-webserver airflow dags trigger ml_train_volatility_models
```

#### Prediction DAG: `ml_predict_volatility`
- **Schedule**: Weekdays at 5 PM (after stock data ETL)
- **Tasks**:
  1. Generate predictions for all tickers
  2. Store predictions in database
  3. Log prediction statistics

**Trigger manually**:
```bash
docker-compose exec airflow-webserver airflow dags trigger ml_predict_volatility
```

## Database Schema

### `ml.volatility_predictions`
Stores model predictions:
```sql
ticker VARCHAR(10)          -- Stock symbol
model_type VARCHAR(20)      -- GARCH or LSTM
forecast_date DATE          -- Prediction date
predicted_volatility NUMERIC  -- Forecasted volatility %
prediction_timestamp TIMESTAMP  -- When prediction was made
```

### `ml.model_metrics`
Stores model performance:
```sql
ticker VARCHAR(10)          -- Stock symbol
model_type VARCHAR(20)      -- GARCH or LSTM
metric_name VARCHAR(50)     -- rmse, mae, mse
metric_value NUMERIC        -- Metric value
run_id VARCHAR(100)         -- MLflow run ID
model_version VARCHAR(20)   -- Model version
created_at TIMESTAMP        -- Timestamp
```

## Getting Started

### 1. Start Services

```bash
# Start all services including MLflow and model API
docker-compose up -d

# Check service health
curl http://localhost:8000/health
curl http://localhost:5000/health
```

### 2. Train Models

```bash
# Option 1: Via Python script (inside container)
docker-compose exec airflow-webserver python /ml_models/train_volatility_models.py

# Option 2: Via Airflow DAG (recommended)
# Go to http://localhost:8080 and trigger ml_train_volatility_models
```

### 3. Generate Predictions

```bash
# Option 1: Via API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model_type": "GARCH", "horizon": 5}'

# Option 2: Via Airflow DAG (recommended)
# Trigger ml_predict_volatility DAG
```

### 4. View Results

- **MLflow UI**: http://localhost:5000 (experiments, models, metrics)
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Dashboard**: http://localhost:8501 → "Volatility Forecast" tab

## Model Training Details

### Training Process

1. **Data Preparation**:
   - Load historical stock prices from PostgreSQL
   - Calculate log returns
   - Normalize data (LSTM only)
   - Split into train/validation/test sets (80/10/10)

2. **Model Training**:
   - GARCH: Fit GARCH(1,1) model to returns
   - LSTM: Train neural network with early stopping

3. **Model Evaluation**:
   - Calculate RMSE, MAE, MSE on test set
   - Generate sample forecasts
   - Log metrics to MLflow

4. **Model Registry**:
   - Save model artifacts
   - Register in MLflow with version
   - Tag with performance metrics

### Hyperparameters

**GARCH**:
- p (GARCH term): 1
- q (ARCH term): 1
- Distribution: Normal

**LSTM**:
- Sequence length: 20 days
- Hidden size: 64
- Number of layers: 2
- Dropout: 0.2
- Learning rate: 0.001
- Batch size: 32
- Epochs: 50-100

## Model Performance

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Measures forecast accuracy
- **MAE** (Mean Absolute Error): Average absolute forecast error
- **MSE** (Mean Squared Error): Squared error (penalizes large errors)

### Typical Performance

| Model | RMSE Range | Best For |
|-------|------------|----------|
| GARCH | 1.5 - 3.0 | Short-term volatility (1-5 days) |
| LSTM | 1.0 - 2.5 | Medium-term patterns (5-10 days) |

*Note: Lower RMSE indicates better performance*

## Production Considerations

### Model Monitoring

Monitor these metrics regularly:
- **Prediction accuracy**: Compare forecasts to realized volatility
- **Model drift**: Track performance degradation over time
- **Data quality**: Ensure input data completeness
- **Inference latency**: API response times

### Model Retraining

Retrain models when:
- Performance degrades beyond threshold (RMSE > 3.0)
- Market regime changes (e.g., high volatility events)
- New data patterns emerge
- Scheduled weekly (automated via DAG)

### Best Practices

1. **Version Control**: All models tracked in MLflow
2. **A/B Testing**: Compare GARCH vs LSTM predictions
3. **Ensemble Methods**: Combine model forecasts for robustness
4. **Feature Engineering**: Add technical indicators, sentiment scores
5. **Backtesting**: Validate on historical out-of-sample data

## API Reference

### POST /predict

Generate volatility predictions.

**Request**:
```json
{
  "ticker": "AAPL",
  "model_type": "GARCH",  // or "LSTM"
  "horizon": 5            // forecast days (1-30)
}
```

**Response**:
```json
{
  "ticker": "AAPL",
  "model_type": "GARCH",
  "horizon": 5,
  "forecasts": [2.5, 2.6, 2.4, 2.5, 2.7],
  "forecast_dates": ["2024-01-15", "2024-01-16", ...],
  "timestamp": "2024-01-14T17:00:00"
}
```

### GET /models

List all registered models.

**Response**:
```json
[
  {
    "model_name": "garch_volatility_aapl",
    "ticker": "AAPL",
    "version": "1",
    "stage": "Production",
    "metrics": {"rmse": 2.1, "mae": 1.8}
  },
  ...
]
```

## Troubleshooting

### Models not training

**Check**:
1. Sufficient data: At least 100 days for GARCH, 200 for LSTM
2. Database connectivity: `docker-compose logs model-api`
3. MLflow service: http://localhost:5000

### Predictions failing

**Check**:
1. Models exist: `curl http://localhost:8000/models`
2. API health: `curl http://localhost:8000/health`
3. Database schema: Ensure `ml` schema exists

### Poor performance

**Solutions**:
1. Increase training data (more historical days)
2. Tune hyperparameters (LSTM epochs, hidden size)
3. Add features (sentiment, technical indicators)
4. Try ensemble methods (combine GARCH + LSTM)

## Development

### Adding New Models

1. Create model class in `models/` directory
2. Implement `train()`, `predict()`, `evaluate()` methods
3. Add to `train_volatility_models.py`
4. Update API endpoints in `api.py`
5. Add visualization to dashboard

### Testing

```bash
# Test model training
python ml_models/train_volatility_models.py --tickers AAPL --models GARCH

# Test API
pytest ml_models/tests/  # (create test suite)

# Test predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model_type": "GARCH", "horizon": 1}'
```

## References

- **GARCH Models**: Engle, R. (1982). Autoregressive Conditional Heteroskedasticity
- **LSTM Networks**: Hochreiter & Schmidhuber (1997). Long Short-Term Memory
- **MLflow**: https://mlflow.org/docs/latest/
- **Financial Volatility**: JP Morgan RiskMetrics

## License

MIT License - See main project LICENSE

## Support

For issues:
1. Check logs: `docker-compose logs mlflow model-api`
2. Verify database: `docker-compose exec postgres psql -U airflow -c "SELECT COUNT(*) FROM ml.volatility_predictions"`
3. Review MLflow UI: http://localhost:5000
