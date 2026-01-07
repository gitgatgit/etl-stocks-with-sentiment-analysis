CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS raw.stock_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.grok_explanations (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    explanation TEXT,
    sentiment VARCHAR(20),
    topic VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE DATABASE metabase;

CREATE TABLE IF NOT EXISTS raw.stock_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.grok_explanations (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    explanation TEXT,
    sentiment VARCHAR(20),
    topic VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
