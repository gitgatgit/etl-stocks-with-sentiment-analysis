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

        # Use yfinance.download() to avoid cookie storage issues
        # This fetches all tickers at once which is more efficient
        try:
            data = yf.download(
                tickers=' '.join(TICKERS),
                period='1d',
                group_by='ticker',
                auto_adjust=False,
                progress=False
            )

            if not data.empty:
                # Handle both single and multiple ticker responses
                if len(TICKERS) == 1:
                    # Single ticker returns a DataFrame
                    ticker = TICKERS[0]
                    if len(data) > 0:
                        row = data.iloc[-1]
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
                        print(f"Inserted data for {ticker}")
                else:
                    # Multiple tickers returns a multi-index DataFrame
                    for ticker in TICKERS:
                        try:
                            if ticker in data.columns.levels[0]:
                                ticker_data = data[ticker]
                                if len(ticker_data) > 0:
                                    row = ticker_data.iloc[-1]
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
                                    print(f"Inserted data for {ticker}")
                        except Exception as e:
                            print(f"Error processing {ticker}: {e}")
            else:
                print("No data returned from yfinance")

        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")
            # Fall back to individual ticker fetching if download fails
            for ticker in TICKERS:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period='1d')

                    if not hist.empty:
                        row = hist.iloc[-1]
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
                        print(f"Inserted data for {ticker} (fallback method)")

                except Exception as e:
                    print(f"Error fetching {ticker} (fallback): {e}")

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
        print(f"Found {len(rows)} stock prices that need explanations")

        if len(rows) == 0:
            print("No stock prices to process. Make sure extract_stock_prices ran successfully.")
            return

        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            print("WARNING: XAI_API_KEY not set. Skipping Grok API calls.")
            return

        client = OpenAI(
            api_key=api_key,
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
