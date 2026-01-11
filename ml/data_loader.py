"""Data loading utilities for ML pipeline."""
import pandas as pd
import psycopg2
from typing import Optional


class StockDataLoader:
    """Load stock and sentiment data from PostgreSQL."""

    def __init__(self, host: str = 'localhost', port: int = 5432,
                 database: str = 'airflow', user: str = 'airflow',
                 password: str = 'airflow'):
        """Initialize database connection parameters."""
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }

    def get_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.conn_params)

    def load_training_data(self, min_date: Optional[str] = None,
                          max_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from analytics.fct_prices_with_grok for training.

        Args:
            min_date: Minimum date to fetch (format: 'YYYY-MM-DD')
            max_date: Maximum date to fetch (format: 'YYYY-MM-DD')

        Returns:
            DataFrame with stock prices and sentiment data
        """
        query = """
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                price_change,
                pct_change,
                sentiment,
                topic
            FROM analytics.fct_prices_with_grok
            WHERE 1=1
        """

        params = []
        if min_date:
            query += " AND date >= %s"
            params.append(min_date)
        if max_date:
            query += " AND date <= %s"
            params.append(max_date)

        query += " ORDER BY ticker, date"

        conn = self.get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params if params else None)
            return df
        finally:
            conn.close()

    def load_raw_stock_data(self, tickers: Optional[list] = None) -> pd.DataFrame:
        """
        Load raw stock data for specific tickers.

        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT'])

        Returns:
            DataFrame with raw stock prices
        """
        query = """
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume
            FROM raw.stock_prices
            WHERE 1=1
        """

        params = []
        if tickers:
            placeholders = ','.join(['%s'] * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        query += " ORDER BY ticker, date"

        conn = self.get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params if params else None)
            return df
        finally:
            conn.close()

    def save_predictions(self, predictions_df: pd.DataFrame, table: str = 'analytics.ml_volatility_predictions'):
        """
        Save model predictions to database.

        Args:
            predictions_df: DataFrame with columns: ticker, date, predicted_volatility_class, predicted_volatility, confidence
            table: Target table name
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Create table if not exists
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    predicted_volatility_class VARCHAR(10) NOT NULL,
                    predicted_volatility NUMERIC,
                    confidence NUMERIC,
                    model_version VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date, model_version)
                )
            """
            cursor.execute(create_table_query)
            conn.commit()

            # Insert predictions
            insert_query = f"""
                INSERT INTO {table}
                (ticker, date, predicted_volatility_class, predicted_volatility, confidence, model_version)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, date, model_version)
                DO UPDATE SET
                    predicted_volatility_class = EXCLUDED.predicted_volatility_class,
                    predicted_volatility = EXCLUDED.predicted_volatility,
                    confidence = EXCLUDED.confidence,
                    created_at = CURRENT_TIMESTAMP
            """

            for _, row in predictions_df.iterrows():
                cursor.execute(insert_query, (
                    row['ticker'],
                    row['date'],
                    row['predicted_volatility_class'],
                    row.get('predicted_volatility'),
                    row.get('confidence'),
                    row.get('model_version', 'v1.0')
                ))

            conn.commit()
            print(f"✓ Saved {len(predictions_df)} predictions to {table}")

        except Exception as e:
            conn.rollback()
            raise Exception(f"Error saving predictions: {e}")
        finally:
            cursor.close()
            conn.close()

    def get_data_statistics(self) -> dict:
        """Get basic statistics about available data."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            stats = {}

            # Count stock prices
            cursor.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM raw.stock_prices")
            count, min_date, max_date = cursor.fetchone()
            stats['stock_prices'] = {
                'count': count,
                'min_date': str(min_date) if min_date else None,
                'max_date': str(max_date) if max_date else None
            }

            # Count by ticker
            cursor.execute("""
                SELECT ticker, COUNT(*) as count
                FROM raw.stock_prices
                GROUP BY ticker
                ORDER BY ticker
            """)
            stats['by_ticker'] = {row[0]: row[1] for row in cursor.fetchall()}

            # Check analytics table
            cursor.execute("SELECT COUNT(*) FROM analytics.fct_prices_with_grok")
            stats['analytics_records'] = cursor.fetchone()[0]

            return stats

        finally:
            cursor.close()
            conn.close()
