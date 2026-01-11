# ETL Stocks with Sentiment Analysis

An ETL pipeline that extracts daily stock prices, enriches them with AI-powered sentiment analysis using xAI's Grok API, and transforms the data for analytics.

## Overview

This pipeline:
- **Extracts** daily stock prices for major tech stocks (AAPL, MSFT, GOOGL, TSLA, NVDA) using yfinance
- **Enriches** data with AI-generated explanations for price movements using xAI Grok
- **Transforms** data using dbt for analytics
- **Predicts** volatility using ML models (GARCH and LSTM) with full MLOps infrastructure
- **Visualizes** results through interactive Streamlit dashboard and Metabase

## Architecture

```
yfinance (Stock Data) → PostgreSQL (raw.stock_prices)
                              ↓
                     Grok API (AI Analysis)
                              ↓
                    PostgreSQL (raw.grok_explanations)
                              ↓
                     dbt (Transform & Test)
                              ↓
                ┌─────────────┴──────────────┐
                ↓                            ↓
       ML Models (GARCH/LSTM)     BI Dashboards (Streamlit + Metabase)
       MLflow + FastAPI
                ↓
    PostgreSQL (ml.predictions)
                ↓
          Dashboard Visualization
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | Apache Airflow 2.8.0 |
| Database | PostgreSQL 14 |
| Data Source | yfinance |
| AI/Sentiment | xAI Grok API |
| ML Models | GARCH (arch), LSTM (PyTorch) |
| MLOps | MLflow 2.10.2 |
| Model Serving | FastAPI |
| Transformation | dbt |
| Visualization | Streamlit, Metabase |
| Runtime | Python 3.11, Docker |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- xAI API key (get one at https://x.ai)

### Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd etl-stocks-with-sentiment-analysis
   ```

2. Create a `.env` file with your API key:
   ```bash
   echo "XAI_API_KEY=your-api-key-here" > .env
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the interfaces:
   - **BI Dashboard**: http://localhost:8501 (Interactive Streamlit dashboard)
   - **Airflow UI**: http://localhost:8080 (admin/admin)
   - **MLflow UI**: http://localhost:5000 (Model tracking and registry)
   - **Model API**: http://localhost:8000 (Volatility predictions API)
   - **Metabase**: http://localhost:3000
   - **PostgreSQL**: localhost:5432 (airflow/airflow)

## CLI Usage

Use the `cli.py` script to manage the pipeline:

```bash
# Run the pipeline with default settings
python cli.py run

# Choose a specific Grok model
python cli.py run --model grok-3
python cli.py run --model grok-3-mini
python cli.py run --model grok-3-fast

# List available Grok models
python cli.py models

# Check pipeline status
python cli.py status

# Trigger the Airflow DAG
python cli.py trigger
```

### Available Grok Models

| Model | Description |
|-------|-------------|
| `grok-3` | Latest full Grok model (default) |
| `grok-3-mini` | Smaller, faster variant |
| `grok-3-fast` | Optimized for speed |

Set the model via environment variable:
```bash
export GROK_MODEL=grok-3-mini
docker-compose up -d
```

Or in your `.env` file:
```
XAI_API_KEY=your-api-key
GROK_MODEL=grok-3
```

## Project Structure

```
etl-stocks-with-sentiment-analysis/
├── airflow/
│   ├── dags/
│   │   ├── stock_grok_pipeline.py         # Main ETL DAG
│   │   ├── ml_train_volatility_models.py  # Model training DAG
│   │   └── ml_predict_volatility.py       # Prediction DAG
│   ├── logs/
│   └── Dockerfile
├── dashboard/                        # Streamlit BI Dashboard
│   ├── app.py                       # Main dashboard application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── ml_models/                        # MLOps Infrastructure
│   ├── models/
│   │   ├── volatility_garch.py      # GARCH model
│   │   └── volatility_lstm.py       # LSTM model
│   ├── train_volatility_models.py   # Training script
│   ├── api.py                       # FastAPI serving
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── dbt_project/
│   ├── models/
│   │   ├── staging/                  # Staging views
│   │   └── marts/                    # Fact tables
│   ├── dbt_project.yml
│   └── profiles.yml
├── cli.py                            # CLI tool
├── docker-compose.yml
├── init.sql                          # Database schema
└── .env                              # Environment variables
```

## Database Schema

### Raw Tables

- `raw.stock_prices` - Daily OHLCV data
- `raw.grok_explanations` - AI-generated insights

### Analytics Tables

- `analytics.stg_stock_prices` - Cleaned prices with calculated fields
- `analytics.stg_grok_explanations` - Normalized explanations
- `analytics.fct_prices_with_grok` - Joined fact table with move categorization

### ML Tables

- `ml.volatility_predictions` - Model predictions (GARCH and LSTM)
- `ml.model_metrics` - Model performance metrics (RMSE, MAE, MSE)

## MLOps Infrastructure

This project includes a complete MLOps pipeline for volatility forecasting:

### Machine Learning Models

1. **GARCH Model**
   - Traditional financial volatility model
   - GARCH(1,1) specification
   - Best for short-term volatility (1-5 days)

2. **LSTM Model**
   - Deep learning approach using PyTorch
   - 2-layer LSTM with 64 hidden units
   - Best for medium-term patterns (5-10 days)

### MLflow Tracking

- **Experiment tracking**: Log parameters, metrics, and artifacts
- **Model registry**: Version control for all models
- **Model comparison**: Compare GARCH vs LSTM performance
- **Access**: http://localhost:5000

### Model Serving API

- **FastAPI endpoint**: Real-time volatility predictions
- **Multi-model support**: GARCH and LSTM
- **Forecast horizon**: 1-30 days ahead
- **Access**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Airflow DAGs

1. **ml_train_volatility_models**
   - Schedule: Weekly (Mondays 2 AM)
   - Trains GARCH and LSTM models for all tickers
   - Evaluates and compares performance
   - Registers models in MLflow

2. **ml_predict_volatility**
   - Schedule: Daily (Weekdays 5 PM)
   - Generates 5-day volatility forecasts
   - Stores predictions in database
   - Updates dashboard visualizations

### Model Performance

Typical RMSE ranges:
- GARCH: 1.5 - 3.0
- LSTM: 1.0 - 2.5

See `ml_models/README.md` for detailed MLOps documentation.

## BI Dashboard Features

The Streamlit dashboard (http://localhost:8501) provides comprehensive visualizations:

### 📊 Overview Tab
- Real-time key metrics and statistics
- Multi-stock performance comparison charts
- Sentiment heatmap across all tickers
- Recent data summary table

### 📈 Stock Analysis Tab
- Interactive candlestick charts with OHLC data
- Volume analysis with color-coded bars
- Daily price change percentage timelines
- Statistical summaries per ticker

### 💭 Sentiment Analysis Tab
- Sentiment distribution pie charts
- Topic breakdown visualizations
- Historical sentiment trends over time
- Sentiment scoring and categorization

### 🔍 AI Explanations Tab
- Complete Grok AI-generated explanations
- Organized by date and ticker
- Price change context for each explanation
- Topic and sentiment tagging

### ⚠️ Large Moves Tab
- Automatic alerts for >5% price movements
- Detailed analysis of each significant move
- AI explanations for volatility
- Visual indicators for trends

### 🔮 Volatility Forecast Tab
- GARCH and LSTM model predictions
- 5-day forward volatility forecasts
- Model comparison visualizations
- Performance metrics (RMSE, MAE)
- Forecast accuracy tracking

**Features:**
- Interactive filters (date range, tickers, sentiment)
- Real-time data updates (5-minute cache)
- Responsive charts with zoom and pan
- Export-ready visualizations
- ML model predictions integration

See `dashboard/README.md` for detailed dashboard documentation.

## Pipeline Schedule

The DAG runs daily at 4 PM EST on weekdays (`0 16 * * 1-5`).

To trigger manually:
```bash
python cli.py trigger
# or via Airflow CLI
docker-compose exec airflow-webserver airflow dags trigger stock_grok_pipeline
```

## Development

### Running dbt locally

```bash
cd dbt_project
dbt run --profiles-dir .
dbt test --profiles-dir .
```

### Debugging

```bash
# Check database contents
python debug_db.py

# View Airflow logs
docker-compose logs -f airflow-scheduler
```

## License

Apache-2.0
