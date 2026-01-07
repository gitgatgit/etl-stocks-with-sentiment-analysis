from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import yfinance as yf
import psycopg2
import os
from openai import OpenAI

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_grok_pipeline',
    default_args=default_args,
    schedule_interval='0 16 * * 1-5',  # Daily at 4pm EST weekdays
    catchup=False,
)

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

def get_db_connection():
    return psycopg2.connect(
        host='postgres',
        database='airflow',
        user='airflow',
        password='airflow'
    )

def extract_stock_prices(**context):
    """Fetch daily stock prices using yfinance"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        execution_date = context['ds']

        for ticker in TICKERS:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1d')

                if not hist.empty:
                    row = hist.iloc[0]
                    cursor.execute("""
                        INSERT INTO raw.stock_prices
                        (ticker, date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        ticker,
                        execution_date,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume'])
                    ))

            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        conn.commit()
    finally:
        cursor.close()
        conn.close()

def call_grok_api(**context):
    """Call Grok API to generate explanations for price moves"""
    import json

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        execution_date = context['ds']

        # Get prices that need explanations
        cursor.execute("""
            SELECT p.ticker, p.date, p.close,
                   LAG(p.close) OVER (PARTITION BY p.ticker ORDER BY p.date) as prev_close
            FROM raw.stock_prices p
            WHERE p.date = %s
            AND NOT EXISTS (
                SELECT 1 FROM raw.grok_explanations g
                WHERE g.ticker = p.ticker AND g.date = p.date
            )
        """, (execution_date,))

        rows = cursor.fetchall()

        client = OpenAI(
            api_key=os.getenv('XAI_API_KEY'),
            base_url="https://api.x.ai/v1"
        )

        for row in rows:
            ticker, date, close, prev_close = row

            if prev_close:
                pct_change = ((close - prev_close) / prev_close) * 100

                prompt = f"""Explain why {ticker} moved {pct_change:.2f}% on {date}.
Provide:
1. A brief 2-sentence explanation
2. Sentiment: positive, negative, or neutral
3. Primary topic: earnings, macro, company-specific, or speculation

Format as JSON: {{"explanation": "...", "sentiment": "...", "topic": "..."}}"""

                try:
                    response = client.chat.completions.create(
                        model="grok-beta",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    result = response.choices[0].message.content

                    try:
                        data = json.loads(result)
                    except json.JSONDecodeError as json_err:
                        print(f"JSON parse error for {ticker}: {json_err}. Response: {result}")
                        continue

                    cursor.execute("""
                        INSERT INTO raw.grok_explanations
                        (ticker, date, explanation, sentiment, topic)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        ticker, date,
                        data.get('explanation', ''),
                        data.get('sentiment', ''),
                        data.get('topic', '')
                    ))

                except Exception as e:
                    print(f"Grok API error for {ticker}: {e}")

        conn.commit()
    finally:
        cursor.close()
        conn.close()

# Tasks
extract_prices = PythonOperator(
    task_id='extract_stock_prices',
    python_callable=extract_stock_prices,
    dag=dag,
)

enrich_with_grok = PythonOperator(
    task_id='call_grok_api',
    python_callable=call_grok_api,
    dag=dag,
)

dbt_run = BashOperator(
    task_id='dbt_transform',
    bash_command='cd /dbt && dbt run --profiles-dir . --log-path /opt/airflow/logs',
    dag=dag,
)

dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command='cd /dbt && dbt test --profiles-dir . --log-path /opt/airflow/logs',
    dag=dag,
)

# Dependencies
extract_prices >> enrich_with_grok >> dbt_run >> dbt_test
