#!/usr/bin/env python3
import psycopg2
import os

# Database connection
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='airflow',
        user='airflow',
        password='airflow'
    )
    cursor = conn.cursor()

    print("=== DATABASE CONNECTION SUCCESSFUL ===\n")

    # Check stock_prices
    cursor.execute("SELECT COUNT(*) FROM raw.stock_prices")
    stock_count = cursor.fetchone()[0]
    print(f"Stock prices in database: {stock_count}")

    cursor.execute("SELECT ticker, date, close FROM raw.stock_prices ORDER BY date DESC LIMIT 5")
    print("\nRecent stock prices:")
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1]} | ${row[2]}")

    # Check grok_explanations
    cursor.execute("SELECT COUNT(*) FROM raw.grok_explanations")
    grok_count = cursor.fetchone()[0]
    print(f"\nGrok explanations in database: {grok_count}")

    if grok_count > 0:
        cursor.execute("SELECT ticker, date, sentiment, topic FROM raw.grok_explanations ORDER BY date DESC LIMIT 5")
        print("\nRecent grok explanations:")
        for row in cursor.fetchall():
            print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")
    else:
        print("  (empty table)")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"Database error: {e}")

# Check environment variable
print("\n=== ENVIRONMENT CHECK ===")
xai_key = os.getenv('XAI_API_KEY')
if xai_key:
    print(f"XAI_API_KEY is set: {xai_key[:10]}..." if len(xai_key) > 10 else "XAI_API_KEY is set (short value)")
else:
    print("XAI_API_KEY is NOT set")
