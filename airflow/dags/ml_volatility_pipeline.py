"""
Airflow DAG for ML Volatility Prediction Pipeline.

This DAG trains and runs volatility prediction models on stock data.
Can be run on-demand or scheduled periodically for model retraining.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys

# Add utils to path for alert imports
sys.path.insert(0, '/opt/airflow')
from airflow.utils.alerts import (
    slack_failure_callback,
    slack_success_callback,
)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': slack_failure_callback,
}


dag = DAG(
    'ml_volatility_pipeline',
    default_args=default_args,
    description='Train and predict stock volatility using ML',
    schedule_interval=None,  # Manual trigger only (or set to '@weekly' for weekly retraining)
    catchup=False,
    tags=['ml', 'volatility', 'prediction'],
)


# Task 1: Check ML dependencies
check_deps = BashOperator(
    task_id='check_ml_dependencies',
    bash_command='python -c "import pandas, numpy, sklearn, xgboost; print(\'✓ All ML dependencies available\')"',
    dag=dag,
)


# Task 2: Train model
train_model = BashOperator(
    task_id='train_volatility_model',
    bash_command='cd /opt/airflow && python -m ml.train --model xgboost --db-host postgres --output-dir /opt/airflow/logs/ml_models',
    dag=dag,
)


# Task 3: Make predictions and save to database
make_predictions = BashOperator(
    task_id='make_predictions',
    bash_command='cd /opt/airflow && python -m ml.predict --save-db --db-host postgres --model /opt/airflow/logs/ml_models/latest.pkl --metadata /opt/airflow/logs/ml_models/latest_metadata.json',
    dag=dag,
)


# Task 4: Validate predictions
validate_predictions = BashOperator(
    task_id='validate_predictions',
    bash_command='''
    python -c "
import psycopg2
conn = psycopg2.connect(host='postgres', database='airflow', user='airflow', password='airflow')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM analytics.ml_volatility_predictions')
count = cursor.fetchone()[0]
print(f'✓ Found {count} predictions in database')
if count == 0:
    raise Exception('No predictions found!')
cursor.close()
conn.close()
"
    ''',
    dag=dag,
    on_success_callback=slack_success_callback,
)


# Set task dependencies
check_deps >> train_model >> make_predictions >> validate_predictions
