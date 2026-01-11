#!/usr/bin/env python3
"""Backfill script to fetch historical stock data and generate Grok explanations."""

import yfinance as yf
import psycopg2
import os
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

def get_db_connection():
    return psycopg2.connect(
        host='localhost',
        port=5432,
        database='airflow',
        user='airflow',
        password='airflow'
    )

def backfill_stock_prices(days=250, start_date=None, end_date=None, clear_existing=True):
    """Fetch historical stock prices for all tickers.

    Args:
        days: Number of trading days to fetch (used if start_date/end_date not provided)
        start_date: Start date string 'YYYY-MM-DD' (optional)
        end_date: End date string 'YYYY-MM-DD' (optional)
        clear_existing: Whether to clear existing data before inserting
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if clear_existing:
        cursor.execute('DELETE FROM raw.grok_explanations')
        print(f"Cleared {cursor.rowcount} grok explanations")
        cursor.execute('DELETE FROM raw.stock_prices')
        print(f"Cleared {cursor.rowcount} stock prices")
        conn.commit()

    # Determine date range
    if start_date and end_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        print(f"\n=== FETCHING STOCK PRICES FROM {start_date} TO {end_date} ===")
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days + 50)  # Extra buffer for weekends/holidays
        print(f"\n=== FETCHING {days} DAYS OF STOCK PRICES ===")

    print(f"Downloading data from {start_dt.date()} to {end_dt.date()}...")

    data = yf.download(
        tickers=' '.join(TICKERS),
        start=start_dt.strftime('%Y-%m-%d'),
        end=end_dt.strftime('%Y-%m-%d'),
        group_by='ticker',
        auto_adjust=False,
        progress=True
    )

    if data.empty:
        print("No data returned from yfinance!")
        return

    inserted = 0
    for ticker in TICKERS:
        ticker_data = data[ticker] if len(TICKERS) > 1 else data
        ticker_data = ticker_data.dropna()

        # Take last N trading days only if using days mode (not explicit date range)
        if not (start_date and end_date):
            ticker_data = ticker_data.tail(days)

        for idx, row in ticker_data.iterrows():
            cursor.execute("""
                INSERT INTO raw.stock_prices
                (ticker, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                ticker,
                idx.strftime('%Y-%m-%d'),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume'])
            ))
            inserted += 1

    conn.commit()
    print(f"Inserted {inserted} stock price records")

    # Verify
    cursor.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM raw.stock_prices")
    count, min_date, max_date = cursor.fetchone()
    print(f"Database now has {count} records from {min_date} to {max_date}")

    cursor.close()
    conn.close()
    return count

def backfill_grok_explanations(limit=None):
    """Generate Grok explanations for price movements."""
    api_key = os.getenv('XAI_API_KEY')
    if not api_key:
        print("XAI_API_KEY not set, skipping Grok explanations")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get prices that need explanations (with previous close for % change)
    query = """
        WITH prices_with_prev AS (
            SELECT ticker, date, close,
                   LAG(close) OVER (PARTITION BY ticker ORDER BY date) as prev_close
            FROM raw.stock_prices
        )
        SELECT ticker, date, close, prev_close
        FROM prices_with_prev
        WHERE prev_close IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM raw.grok_explanations g
            WHERE g.ticker = prices_with_prev.ticker AND g.date = prices_with_prev.date
        )
        ORDER BY date DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()

    print(f"\n=== GENERATING GROK EXPLANATIONS FOR {len(rows)} RECORDS ===")

    if not rows:
        print("No records need explanations")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )
    grok_model = os.getenv('GROK_MODEL', 'grok-4-1-fast-reasoning')

    for i, (ticker, date, close, prev_close) in enumerate(rows):
        pct_change = ((close - prev_close) / prev_close) * 100

        prompt = f"""Explain why {ticker} moved {pct_change:.2f}% on {date}.
Provide:
1. A brief 2-sentence explanation
2. Sentiment: positive, negative, or neutral
3. Primary topic: earnings, macro, company-specific, or speculation

Format as JSON: {{"explanation": "...", "sentiment": "...", "topic": "..."}}"""

        try:
            response = client.chat.completions.create(
                model=grok_model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content
            data = json.loads(result)

            cursor.execute("""
                INSERT INTO raw.grok_explanations
                (ticker, date, explanation, sentiment, topic)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                ticker, date,
                data.get('explanation', ''),
                data.get('sentiment', ''),
                data.get('topic', '')
            ))
            conn.commit()
            print(f"[{i+1}/{len(rows)}] {ticker} {date}: {pct_change:+.2f}% - {data.get('sentiment')}")

        except json.JSONDecodeError as e:
            print(f"[{i+1}/{len(rows)}] {ticker} {date}: JSON parse error - {e}")
        except Exception as e:
            print(f"[{i+1}/{len(rows)}] {ticker} {date}: API error - {e}")

        # Rate limiting
        time.sleep(0.5)

    cursor.execute("SELECT COUNT(*) FROM raw.grok_explanations")
    count = cursor.fetchone()[0]
    print(f"\nTotal Grok explanations in database: {count}")

    cursor.close()
    conn.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backfill stock data and Grok explanations')
    parser.add_argument('--days', type=int, default=250, help='Number of trading days to fetch (ignored if --start-date/--end-date provided)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear existing data before inserting')
    parser.add_argument('--grok-limit', type=int, default=None, help='Limit Grok API calls (None=all)')
    parser.add_argument('--skip-grok', action='store_true', help='Skip Grok explanations')
    args = parser.parse_args()

    # Validate date arguments
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        parser.error('--start-date and --end-date must be used together')

    backfill_stock_prices(
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        clear_existing=not args.no_clear
    )

    if not args.skip_grok:
        backfill_grok_explanations(args.grok_limit)
