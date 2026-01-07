{{ config(materialized='table') }}

SELECT
    p.ticker,
    p.date,
    p.open,
    p.high,
    p.low,
    p.close,
    p.volume,
    p.price_change,
    p.pct_change,
    g.explanation,
    g.sentiment,
    g.topic,
    CASE 
        WHEN ABS(p.pct_change) > 5 THEN 'large_move'
        WHEN ABS(p.pct_change) > 2 THEN 'medium_move'
        ELSE 'small_move'
    END as move_category
FROM {{ ref('stg_stock_prices') }} p
LEFT JOIN {{ ref('stg_grok_explanations') }} g
  ON p.ticker = g.ticker
 AND p.date = g.date
