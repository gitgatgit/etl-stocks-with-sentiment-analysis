{{ config(materialized='view') }}

SELECT
    ticker,
    date,
    open,
    high,
    low,
    close,
    volume,
    close - LAG(close) OVER (PARTITION BY ticker ORDER BY date) as price_change,
    ((close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) 
     / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) * 100 as pct_change
FROM {{ source('raw', 'stock_prices') }}
