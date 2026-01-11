from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta
import json
import os

# DAG configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_predict_volatility',
    default_args=default_args,
    description='Generate daily volatility predictions using trained models',
    schedule_interval='0 17 * * 1-5',  # Weekdays at 5 PM (after stock data ETL)
    catchup=False,
    tags=['ml', 'volatility', 'prediction']
)


def get_tickers():
    """Get list of tickers from database"""
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASS', 'airflow')
    )

    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM raw.stock_prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return tickers


def generate_predictions(**context):
    """Generate predictions for all tickers using both models"""
    import requests
    import pandas as pd

    tickers = get_tickers()
    model_types = ['GARCH', 'LSTM']
    horizon = 5  # 5-day forecast

    results = []

    for ticker in tickers:
        for model_type in model_types:
            try:
                # Call prediction API
                response = requests.post(
                    'http://model-api:8000/predict',
                    json={
                        'ticker': ticker,
                        'model_type': model_type,
                        'horizon': horizon
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        'ticker': ticker,
                        'model_type': model_type,
                        'success': True,
                        'forecasts': result.get('forecasts', [])
                    })
                    print(f"✓ Generated {model_type} predictions for {ticker}")
                else:
                    print(f"✗ Failed to generate {model_type} predictions for {ticker}: {response.status_code}")
                    results.append({
                        'ticker': ticker,
                        'model_type': model_type,
                        'success': False,
                        'error': response.text
                    })

            except Exception as e:
                print(f"✗ Error predicting {ticker} with {model_type}: {str(e)}")
                results.append({
                    'ticker': ticker,
                    'model_type': model_type,
                    'success': False,
                    'error': str(e)
                })

    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"\nPrediction Summary: {successful}/{total} successful")

    # Push results to XCom
    context['task_instance'].xcom_push(key='prediction_results', value=results)

    return results


def log_prediction_stats(**context):
    """Log prediction statistics to database"""
    import psycopg2
    from datetime import datetime

    results = context['task_instance'].xcom_pull(
        task_ids='generate_predictions',
        key='prediction_results'
    )

    if not results:
        print("No results to log")
        return

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASS', 'airflow')
    )

    cursor = conn.cursor()

    # Count predictions by model type
    for result in results:
        if result['success']:
            ticker = result['ticker']
            model_type = result['model_type']
            num_forecasts = len(result.get('forecasts', []))

            cursor.execute("""
                INSERT INTO ml.model_metrics
                (ticker, model_type, metric_name, metric_value)
                VALUES (%s, %s, %s, %s)
            """, (ticker, model_type, 'predictions_generated', num_forecasts))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Logged prediction statistics for {len(results)} model runs")


# Task 1: Generate predictions
predict_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag,
    provide_context=True
)

# Task 2: Log statistics
log_stats_task = PythonOperator(
    task_id='log_prediction_stats',
    python_callable=log_prediction_stats,
    dag=dag,
    provide_context=True
)

# Set dependencies
predict_task >> log_stats_task
