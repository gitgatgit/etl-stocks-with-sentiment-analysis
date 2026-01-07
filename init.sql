CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

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

-- Create metabase database (will fail if already exists, which is fine)
CREATE DATABASE metabase;
