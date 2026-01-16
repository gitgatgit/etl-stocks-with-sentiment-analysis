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
| MLOps | MLflow 2.10+ (Experiment Tracking & Model Registry) |
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
   - **MLflow UI**: http://localhost:5000
   - **Metabase**: http://localhost:3000
   - **Streamlit Dashboard**: http://localhost:8501
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

## MLflow Experiment Tracking

MLflow provides comprehensive experiment tracking, model registry, and model versioning capabilities.

### Accessing MLflow UI

Access the MLflow UI at **http://localhost:5000** to:
- View all experiment runs with metrics and parameters
- Compare model performance across runs
- Browse model artifacts and feature importance
- Manage the model registry (Staging, Production versions)

### Training with MLflow

Models are automatically tracked in MLflow when training:

```bash
# Train with MLflow tracking (default)
python cli.py ml-train --ml-model xgboost

# Disable MLflow tracking
python ml/train.py --no-mlflow

# Custom experiment name
python ml/train.py --mlflow-experiment my-experiment
```

### What MLflow Tracks

**Parameters:**
- Model type and hyperparameters
- Dataset split sizes
- Number of features
- Random seed

**Metrics:**
- Accuracy, F1 scores (macro/weighted), Recall (macro/weighted)
- Separate metrics for validation and test sets
- Confusion matrix

**Artifacts:**
- Trained model (in sklearn format)
- Feature importance CSV
- Confusion matrix JSON
- Model signature and input examples

### Using the Model Registry

Models are automatically registered as `volatility_predictor`:

```bash
# View registered models
curl http://localhost:5000/api/2.0/mlflow/registered-models/list

# Load production model in Python
from ml.mlflow_utils import MLflowTracker

tracker = MLflowTracker()
model = tracker.load_model("models:/volatility_predictor/Production")
```

### Comparing Experiments

Use the MLflow UI to:
1. Navigate to the "volatility-prediction" experiment
2. Select multiple runs
3. Click "Compare" to view side-by-side metrics
4. Identify the best performing model

### MLflow Backend Storage

- **Tracking Server**: http://mlflow:5000 (internal), http://localhost:5000 (external)
- **Backend Store**: PostgreSQL database (metrics and parameters)
- **Artifact Store**: Docker volume `mlflow-artifacts` (models and files)

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

## CI/CD and Testing

### GitHub Actions Pipelines

This project includes comprehensive CI/CD automation:

#### **CI Pipeline** (`.github/workflows/ci.yml`)
Runs automatically on every push and pull request:

- **Code Quality Checks**: Black, isort, Flake8, Pylint, MyPy
- **Unit Tests**: Pytest with coverage reporting
- **Integration Tests**: Database integration tests with PostgreSQL
- **DAG Validation**: Airflow DAG structure and import validation
- **Security Scanning**: Dependency vulnerability checks, secret detection
- **Build Testing**: Docker Compose validation

#### **CD Pipeline** (`.github/workflows/cd.yml`)
Handles deployment to staging and production:

- **Build & Push**: Docker images to GitHub Container Registry
- **Staging Deployment**: Automatic deployment on main branch
- **Production Deployment**: Manual approval for production releases
- **Database Backups**: Pre-deployment backups
- **Rollback Support**: Automatic rollback on deployment failures

### Local Testing

#### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

#### Run Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=ml --cov-report=html

# Run specific test file
pytest tests/test_feature_engineering.py -v
```

#### Run Linting

```bash
# Format code
black .
isort .

# Check code quality
flake8 .
pylint ml/

# Type checking
mypy ml/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Pre-commit will automatically run:
- Code formatting (Black, isort)
- Linting (Flake8)
- Security checks (Bandit)
- Type checking (MyPy)
- SQL formatting (SQLFluff)

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures
├── test_feature_engineering.py # Feature engineering tests
├── test_data_loader.py         # Data loader tests
└── test_airflow_dags.py        # Airflow DAG validation tests
```

### CI/CD Configuration Files

- **`.github/workflows/ci.yml`** - Continuous Integration pipeline
- **`.github/workflows/cd.yml`** - Continuous Deployment pipeline
- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration
- **`pytest.ini`** - Pytest configuration
- **`pyproject.toml`** - Python tooling configuration
- **`requirements-dev.txt`** - Development dependencies

## License

MIT
