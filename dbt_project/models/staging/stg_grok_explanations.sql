{{ config(materialized='view') }}

SELECT
    ticker,
    date,
    explanation,
    LOWER(sentiment) as sentiment,
    LOWER(topic) as topic
FROM {{ source('raw', 'grok_explanations') }}
