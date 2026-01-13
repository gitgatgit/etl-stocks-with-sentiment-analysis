-- Dashboard views for Metabase
-- Run this to recreate views after database reset

-- Actual volatility calculated from price data
CREATE OR REPLACE VIEW analytics.actual_volatility AS
SELECT
    ticker,
    date,
    (high - low) / NULLIF(close, 0) * 100 AS daily_volatility_pct,
    CASE
        WHEN (high - low) / NULLIF(close, 0) * 100 < 2 THEN 'low'
        WHEN (high - low) / NULLIF(close, 0) * 100 < 5 THEN 'medium'
        ELSE 'high'
    END AS actual_volatility_class
FROM raw.stock_prices
WHERE close > 0;

-- Prediction accuracy comparison
CREATE OR REPLACE VIEW analytics.prediction_accuracy AS
SELECT
    p.ticker,
    p.date AS prediction_date,
    p.predicted_volatility_class,
    a.actual_volatility_class,
    p.confidence,
    p.model_version,
    p.created_at AS predicted_at,
    CASE
        WHEN p.predicted_volatility_class = a.actual_volatility_class THEN 1
        ELSE 0
    END AS is_correct,
    CASE
        WHEN p.predicted_volatility_class = a.actual_volatility_class THEN '✅ Correct'
        ELSE '❌ Wrong'
    END AS result
FROM analytics.ml_volatility_predictions p
JOIN analytics.actual_volatility a
    ON p.ticker = a.ticker
    AND p.date = a.date;

-- Accuracy by stock
CREATE OR REPLACE VIEW analytics.accuracy_by_stock AS
SELECT
    ticker AS stock,
    COUNT(*) AS total_predictions,
    SUM(is_correct) AS correct,
    ROUND(AVG(is_correct) * 100, 1) || '%' AS accuracy,
    ROUND(AVG(confidence) * 100, 1) || '%' AS avg_confidence
FROM analytics.prediction_accuracy
GROUP BY ticker
ORDER BY AVG(is_correct) DESC;

-- Accuracy by day
CREATE OR REPLACE VIEW analytics.accuracy_by_day AS
SELECT
    prediction_date AS date,
    COUNT(*) AS predictions,
    SUM(is_correct) AS correct,
    ROUND(AVG(is_correct) * 100, 0) || '%' AS accuracy,
    STRING_AGG(
        CASE WHEN is_correct = 1 THEN '✅' ELSE '❌' END,
        ' ' ORDER BY ticker
    ) AS results
FROM analytics.prediction_accuracy
GROUP BY prediction_date
ORDER BY prediction_date DESC;

-- Overall model performance summary
CREATE OR REPLACE VIEW analytics.model_performance_summary AS
SELECT
    COUNT(*) AS total_predictions,
    SUM(is_correct) AS correct_predictions,
    ROUND(AVG(is_correct) * 100, 1) AS overall_accuracy_pct,
    ROUND(AVG(confidence) * 100, 1) AS avg_confidence_pct,
    COUNT(DISTINCT ticker) AS stocks_tracked,
    MIN(prediction_date) AS tracking_since,
    MAX(prediction_date) AS latest_prediction
FROM analytics.prediction_accuracy;

-- Today's predictions dashboard
CREATE OR REPLACE VIEW analytics.volatility_dashboard AS
SELECT
    ticker AS stock,
    date AS prediction_date,
    CASE predicted_volatility_class
        WHEN 'low' THEN '🟢 Low Risk'
        WHEN 'medium' THEN '🟡 Medium Risk'
        WHEN 'high' THEN '🔴 High Risk'
    END AS expected_volatility,
    ROUND(confidence * 100, 0)::text || '%' AS confidence,
    CASE
        WHEN confidence >= 0.8 THEN 'Strong'
        WHEN confidence >= 0.6 THEN 'Moderate'
        ELSE 'Weak'
    END AS signal_strength,
    created_at::date AS generated_on
FROM analytics.ml_volatility_predictions
WHERE model_version = (
    SELECT model_version
    FROM analytics.ml_volatility_predictions
    ORDER BY created_at DESC
    LIMIT 1
);

-- Accuracy by model version
CREATE OR REPLACE VIEW analytics.accuracy_by_model AS
SELECT
    model_version AS model,
    COUNT(*) AS predictions,
    SUM(is_correct) AS correct,
    ROUND(AVG(is_correct) * 100, 1) || '%' AS accuracy,
    MIN(predicted_at::date) AS first_used,
    MAX(predicted_at::date) AS last_used
FROM analytics.prediction_accuracy
GROUP BY model_version
ORDER BY MAX(predicted_at) DESC;

-- Recent predictions with results
CREATE OR REPLACE VIEW analytics.recent_predictions_results AS
SELECT
    ticker AS stock,
    prediction_date AS date,
    CASE predicted_volatility_class
        WHEN 'low' THEN '🟢 Low'
        WHEN 'medium' THEN '🟡 Medium'
        WHEN 'high' THEN '🔴 High'
    END AS predicted,
    CASE actual_volatility_class
        WHEN 'low' THEN '🟢 Low'
        WHEN 'medium' THEN '🟡 Medium'
        WHEN 'high' THEN '🔴 High'
    END AS actual,
    result,
    ROUND(confidence * 100, 0) || '%' AS confidence
FROM analytics.prediction_accuracy
ORDER BY prediction_date DESC, ticker;

-- Confusion matrix
CREATE OR REPLACE VIEW analytics.prediction_confusion_matrix AS
SELECT
    predicted_volatility_class AS predicted,
    actual_volatility_class AS actual,
    COUNT(*) AS count
FROM analytics.prediction_accuracy
GROUP BY predicted_volatility_class, actual_volatility_class
ORDER BY predicted_volatility_class, actual_volatility_class;

-- Precision per class
CREATE OR REPLACE VIEW analytics.model_precision AS
SELECT
    predicted_volatility_class as class,
    SUM(CASE WHEN predicted_volatility_class = actual_volatility_class THEN 1 ELSE 0 END) as true_positives,
    COUNT(*) as total_predicted,
    ROUND(100.0 * SUM(CASE WHEN predicted_volatility_class = actual_volatility_class THEN 1 ELSE 0 END) / COUNT(*), 1) as precision_pct
FROM analytics.prediction_accuracy
GROUP BY predicted_volatility_class
ORDER BY predicted_volatility_class;

-- Recall per class
CREATE OR REPLACE VIEW analytics.model_recall AS
SELECT
    actual_volatility_class as class,
    SUM(CASE WHEN predicted_volatility_class = actual_volatility_class THEN 1 ELSE 0 END) as true_positives,
    COUNT(*) as total_actual,
    ROUND(100.0 * SUM(CASE WHEN predicted_volatility_class = actual_volatility_class THEN 1 ELSE 0 END) / COUNT(*), 1) as recall_pct
FROM analytics.prediction_accuracy
GROUP BY actual_volatility_class
ORDER BY actual_volatility_class;

-- Combined metrics: Precision, Recall, F1
CREATE OR REPLACE VIEW analytics.model_metrics AS
SELECT
    p.class,
    p.true_positives,
    p.total_predicted,
    r.total_actual,
    p.precision_pct,
    r.recall_pct,
    ROUND(2.0 * (p.precision_pct * r.recall_pct) / NULLIF(p.precision_pct + r.recall_pct, 0), 1) as f1_score
FROM analytics.model_precision p
JOIN analytics.model_recall r ON p.class = r.class
ORDER BY p.class;
