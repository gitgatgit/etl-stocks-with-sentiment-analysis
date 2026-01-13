# ETL Stocks with Sentiment Analysis

An ETL pipeline that extracts daily stock prices, enriches them with AI-powered sentiment analysis using xAI's Grok API, and transforms the data for analytics.

## Overview

This pipeline:
- **Extracts** daily stock prices for major tech stocks (AAPL, MSFT, GOOGL, TSLA, NVDA) using yfinance
- **Enriches** data with AI-generated explanations for price movements using xAI Grok
- **Transforms** data using dbt for analytics
- **Predicts** next-day volatility using machine learning models
- **Visualizes** results through Streamlit and Metabase dashboards

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
                    PostgreSQL (analytics.fct_prices_with_grok)
                              ↓
                  ┌───────────┼───────────┐
                  │           │           │
         ML Models    Streamlit    Metabase
         (Volatility) (Dashboard)  (Dashboards)
                  │
    PostgreSQL (ml_volatility_predictions)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | Apache Airflow 2.8.0 |
| Database | PostgreSQL 14 |
| Data Source | yfinance |
| AI Analysis | xAI Grok API |
| ML Models | XGBoost, Random Forest, scikit-learn |
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
   - **Airflow UI**: http://localhost:8080 (admin/admin)
   - **Metabase**: http://localhost:3000
   - **PostgreSQL**: localhost:5432 (airflow/airflow)

## CLI Usage

Use the `cli.py` script to manage the pipeline:

```bash
# Run the pipeline with default settings
python cli.py run

# Choose a specific Grok model
python cli.py run --model grok-4-1-fast-reasoning
python cli.py run --model grok-4-fast-reasoning

# List available Grok models
python cli.py models

# Check pipeline status
python cli.py status

# Trigger the Airflow DAG
python cli.py trigger

# Train ML volatility prediction model
python cli.py ml-train

# Make volatility predictions
python cli.py ml-predict --save-db
```

### ML Commands

```bash
# Train a volatility prediction model
python cli.py ml-train --ml-model xgboost

# Train with specific date range
python cli.py ml-train --min-date 2024-01-01 --max-date 2024-12-31

# Make predictions and save to database
python cli.py ml-predict --save-db

# Predict for specific tickers
python cli.py ml-predict --tickers AAPL MSFT --save-db

# Export predictions to CSV
python cli.py ml-predict --output predictions.csv
```

See [`ml/README.md`](ml/README.md) for detailed ML documentation.

### Available Grok Models

| Model | Description |
|-------|-------------|
| `grok-4-1-fast-reasoning` | Latest Grok 4.1 with fast reasoning (default) |
| `grok-4-fast-reasoning` | Grok 4 with fast reasoning |

Set the model via environment variable:
```bash
export GROK_MODEL=grok-4-fast-reasoning
docker-compose up -d
```

Or in your `.env` file:
```
XAI_API_KEY=your-api-key
GROK_MODEL=grok-4-1-fast-reasoning
```

## Project Structure

```
etl-stocks-with-sentiment-analysis/
├── airflow/
│   ├── dags/
│   │   ├── stock_grok_pipeline.py   # Main ETL DAG
│   │   └── ml_volatility_pipeline.py # ML training DAG
│   ├── logs/
│   └── Dockerfile
├── dbt_project/
│   ├── models/
│   │   ├── staging/                  # Staging views
│   │   └── marts/                    # Fact tables
│   ├── dbt_project.yml
│   └── profiles.yml
├── ml/
│   ├── data_loader.py                # Database utilities
│   ├── feature_engineering.py        # Technical indicators & features
│   ├── train.py                      # Model training script
│   ├── predict.py                    # Prediction script
│   ├── models/                       # Trained models
│   ├── requirements.txt              # ML dependencies
│   └── README.md                     # ML documentation
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
- `analytics.ml_volatility_predictions` - ML model predictions for next-day volatility

## Pipeline Schedule

The DAG runs daily at 4 PM EST on weekdays (`0 16 * * 1-5`).

To trigger manually:
```bash
python cli.py trigger
# or via Airflow CLI
docker-compose exec airflow-webserver airflow dags trigger stock_grok_pipeline
```

## ML Volatility Prediction

The ML module predicts **next-day volatility** for stocks by classifying them into three categories:
- **Low volatility**: < 2% intraday range
- **Medium volatility**: 2% - 5% intraday range
- **High volatility**: > 5% intraday range

### Features

- **Technical Indicators**: RSI, ATR, Bollinger Bands, price returns
- **Historical Statistics**: Rolling volatility, volume ratios
- **Sentiment Analysis**: Grok-generated sentiment and topics
- **Temporal Features**: Day of week, month

### Models

- **XGBoost** (recommended): 200 estimators, max depth 6
- **Random Forest**: 200 estimators, max depth 10

### Quick Start

```bash
# Install ML dependencies
pip install -r ml/requirements.txt

# Train a model
python cli.py ml-train --ml-model xgboost

# Make predictions
python cli.py ml-predict --save-db
```

For detailed documentation, see [`ml/README.md`](ml/README.md).

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

MIT
