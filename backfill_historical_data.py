#!/usr/bin/env python3
"""
Backfill historical stock data for ML model training

This script loads historical stock prices without calling the Grok API
to quickly accumulate data for volatility model training.
"""

import yfinance as yf
import psycopg2
from datetime import datetime, timedelta
import os

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
DAYS_BACK = 365  # Load 1 year of data (adjust as needed: 100, 250, 365, 730)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASS', 'airflow')
    )

def backfill_stock_prices(days_back=365):
    """Load historical stock prices"""
    print(f"🔄 Backfilling {days_back} days of stock data for {len(TICKERS)} tickers...")

    conn = get_db_connection()
    cursor = conn.cursor()

    total_inserted = 0

    for ticker in TICKERS:
        print(f"\n📈 Processing {ticker}...")

        try:
            # Download historical data
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                print(f"   ⚠️  No data found for {ticker}")
                continue

            print(f"   📊 Found {len(df)} days of data")

            # Insert into database
            inserted = 0
            for date, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO raw.stock_prices
                        (ticker, date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO NOTHING
                    """, (
                        ticker,
                        date.date(),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume'])
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"   ❌ Error inserting {ticker} on {date.date()}: {e}")

            conn.commit()
            total_inserted += inserted
            print(f"   ✅ Inserted {inserted} records for {ticker}")

        except Exception as e:
            print(f"   ❌ Error processing {ticker}: {e}")
            continue

    cursor.close()
    conn.close()

    print(f"\n{'='*60}")
    print(f"✅ Backfill complete!")
    print(f"📊 Total records inserted: {total_inserted}")
    print(f"{'='*60}")

    return total_inserted

def check_data_status():
    """Check current data in database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("📊 DATA STATUS CHECK")
    print("="*60)

    for ticker in TICKERS:
        cursor.execute("""
            SELECT COUNT(*), MIN(date), MAX(date)
            FROM raw.stock_prices
            WHERE ticker = %s
        """, (ticker,))

        count, min_date, max_date = cursor.fetchone()

        status = "✅" if count >= 100 else "⚠️"
        print(f"{status} {ticker}: {count} days ({min_date} to {max_date})")

    cursor.execute("SELECT COUNT(*) FROM raw.stock_prices")
    total = cursor.fetchone()[0]

    print("="*60)
    print(f"Total records: {total}")

    # Check if ready for training
    cursor.execute("""
        SELECT ticker, COUNT(*) as days
        FROM raw.stock_prices
        GROUP BY ticker
        HAVING COUNT(*) >= 100
    """)

    ready_for_garch = cursor.fetchall()

    cursor.execute("""
        SELECT ticker, COUNT(*) as days
        FROM raw.stock_prices
        GROUP BY ticker
        HAVING COUNT(*) >= 200
    """)

    ready_for_lstm = cursor.fetchall()

    print(f"\n🎯 Ready for GARCH training (≥100 days): {len(ready_for_garch)} tickers")
    print(f"🎯 Ready for LSTM training (≥200 days): {len(ready_for_lstm)} tickers")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Backfill historical stock data')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days to backfill (default: 365)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check current status, do not backfill')

    args = parser.parse_args()

    if args.check_only:
        check_data_status()
    else:
        print(f"🚀 Starting backfill for {args.days} days...")
        print(f"⏳ This may take a few minutes...\n")

        backfill_stock_prices(args.days)

        print("\n📊 Checking final status...")
        check_data_status()

        print("\n" + "="*60)
        print("💡 NEXT STEPS:")
        print("="*60)
        print("1. If you have ≥100 days: Train GARCH models")
        print("   python ml_models/train_volatility_models.py --models GARCH")
        print()
        print("2. If you have ≥200 days: Train both models")
        print("   python ml_models/train_volatility_models.py --models GARCH LSTM")
        print()
        print("3. View data in dashboard:")
        print("   http://localhost:8501")
        print("="*60)
