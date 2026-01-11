# ETL Stocks with Sentiment Analysis

An ETL pipeline that extracts daily stock prices, enriches them with AI-powered sentiment analysis using xAI's Grok API, and transforms the data for analytics.

## Overview

This pipeline:
- **Extracts** daily stock prices for major tech stocks (AAPL, MSFT, GOOGL, TSLA, NVDA) using yfinance
- **Enriches** data with AI-generated explanations for price movements using xAI Grok
- **Transforms** data using dbt for analytics
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
              BI Dashboards (Streamlit + Metabase)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | Apache Airflow 2.8.0 |
| Database | PostgreSQL 14 |
| Data Source | yfinance |
| AI/ML | xAI Grok API |
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
│   │   └── stock_grok_pipeline.py   # Main ETL DAG
│   ├── logs/
│   └── Dockerfile
├── dashboard/                        # Streamlit BI Dashboard
│   ├── app.py                       # Main dashboard application
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

**Features:**
- Interactive filters (date range, tickers, sentiment)
- Real-time data updates (5-minute cache)
- Responsive charts with zoom and pan
- Export-ready visualizations

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
