from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# DAG configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_train_volatility_models',
    default_args=default_args,
    description='Train volatility forecasting models (GARCH and LSTM)',
    schedule_interval='0 2 * * 1',  # Weekly on Monday at 2 AM
    catchup=False,
    tags=['ml', 'volatility', 'training']
)

# Task 1: Train GARCH models
train_garch = BashOperator(
    task_id='train_garch_models',
    bash_command="""
    cd /ml_models && \
    python train_volatility_models.py --models GARCH
    """,
    dag=dag,
    env={
        'DB_HOST': 'postgres',
        'DB_PORT': '5432',
        'DB_NAME': 'airflow',
        'DB_USER': 'airflow',
        'DB_PASS': 'airflow',
        'MLFLOW_TRACKING_URI': 'http://mlflow:5000'
    }
)

# Task 2: Train LSTM models
train_lstm = BashOperator(
    task_id='train_lstm_models',
    bash_command="""
    cd /ml_models && \
    python train_volatility_models.py --models LSTM --epochs 100
    """,
    dag=dag,
    env={
        'DB_HOST': 'postgres',
        'DB_PORT': '5432',
        'DB_NAME': 'airflow',
        'DB_USER': 'airflow',
        'DB_PASS': 'airflow',
        'MLFLOW_TRACKING_URI': 'http://mlflow:5000'
    }
)

# Task 3: Evaluate models
def evaluate_models():
    """Compare model performances and log results"""
    import psycopg2
    import pandas as pd
    from mlflow import MlflowClient

    # Connect to MLflow
    client = MlflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

    # Get experiment
    experiment = client.get_experiment_by_name("volatility-forecasting")

    if experiment:
        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100
        )

        # Extract metrics
        results = []
        for run in runs:
            ticker = run.data.params.get('ticker', 'UNKNOWN')
            model_type = run.data.params.get('model_type', 'UNKNOWN')
            rmse = run.data.metrics.get('test_rmse', None)

            if rmse:
                results.append({
                    'ticker': ticker,
                    'model_type': model_type,
                    'rmse': rmse,
                    'run_id': run.info.run_id
                })

        # Store metrics in database
        if results:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'postgres'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'airflow'),
                user=os.getenv('DB_USER', 'airflow'),
                password=os.getenv('DB_PASS', 'airflow')
            )
            cursor = conn.cursor()

            for result in results:
                cursor.execute("""
                    INSERT INTO ml.model_metrics
                    (ticker, model_type, metric_name, metric_value, run_id)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    result['ticker'],
                    result['model_type'],
                    'rmse',
                    result['rmse'],
                    result['run_id']
                ))

            conn.commit()
            cursor.close()
            conn.close()

            print(f"Stored {len(results)} model metrics")

            # Print summary
            df = pd.DataFrame(results)
            print("\nModel Performance Summary:")
            print(df.groupby(['ticker', 'model_type'])['rmse'].min())

evaluate_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag
)

# Set dependencies
train_garch >> evaluate_task
train_lstm >> evaluate_task
