CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS raw.stock_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS raw.grok_explanations (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    explanation TEXT,
    sentiment VARCHAR(20),
    topic VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- ML tables for volatility predictions
CREATE TABLE IF NOT EXISTS ml.volatility_predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    model_type VARCHAR(20) NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_volatility NUMERIC,
    prediction_timestamp TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, model_type, forecast_date)
);

CREATE INDEX IF NOT EXISTS idx_vol_pred_ticker_date
ON ml.volatility_predictions(ticker, forecast_date);

CREATE TABLE IF NOT EXISTS ml.model_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    model_type VARCHAR(20) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value NUMERIC,
    run_id VARCHAR(100),
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_ticker
ON ml.model_metrics(ticker, model_type, created_at DESC);

-- Create metabase database (will fail if already exists, which is fine)
CREATE DATABASE metabase;
