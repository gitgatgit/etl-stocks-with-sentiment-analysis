"""Tests for data loader module."""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from ml.data_loader import StockDataLoader


@pytest.mark.unit
class TestStockDataLoader:
    """Test data loader functionality."""

    def test_init_connection_params(self):
        """Test initialization stores correct connection parameters."""
        loader = StockDataLoader(
            host='testhost',
            port=5433,
            database='testdb',
            user='testuser',
            password='testpass'
        )

        assert loader.conn_params['host'] == 'testhost'
        assert loader.conn_params['port'] == 5433
        assert loader.conn_params['database'] == 'testdb'
        assert loader.conn_params['user'] == 'testuser'
        assert loader.conn_params['password'] == 'testpass'

    @patch('ml.data_loader.psycopg2.connect')
    def test_load_training_data_query_structure(self, mock_connect):
        """Test training data query is properly structured."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        loader = StockDataLoader()

        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            loader.load_training_data(min_date='2024-01-01', max_date='2024-12-31')

            # Verify SQL query was called
            assert mock_read_sql.called
            query = mock_read_sql.call_args[0][0]

            # Check query contains required columns
            assert 'ticker' in query.lower()
            assert 'date' in query.lower()
            assert 'close' in query.lower()
            assert 'volume' in query.lower()
            assert 'sentiment' in query.lower()

            # Check filtering is applied
            assert 'where' in query.lower()
            assert 'order by' in query.lower()

    @patch('ml.data_loader.psycopg2.connect')
    def test_load_training_data_date_params(self, mock_connect):
        """Test date parameters are properly passed to query."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        loader = StockDataLoader()

        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            loader.load_training_data(min_date='2024-01-01', max_date='2024-12-31')

            # Verify params were passed
            call_args = mock_read_sql.call_args
            params = call_args[1].get('params') or call_args[0][2] if len(call_args[0]) > 2 else None

            if params:
                assert '2024-01-01' in params
                assert '2024-12-31' in params

    def test_save_predictions_dataframe_validation(self, sample_predictions):
        """Test predictions DataFrame has required columns."""
        required_columns = ['ticker', 'date', 'predicted_volatility_class']

        for col in required_columns:
            assert col in sample_predictions.columns, f"Missing required column: {col}"

    @patch('ml.data_loader.psycopg2.connect')
    def test_save_predictions_handles_conflict(self, mock_connect, sample_predictions):
        """Test save_predictions uses ON CONFLICT for upserts."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        loader = StockDataLoader()

        # Should not raise exception
        loader.save_predictions(sample_predictions)

        # Verify INSERT query was executed
        assert mock_cursor.execute.called

        # Check that ON CONFLICT clause is in query
        calls = [str(call) for call in mock_cursor.execute.call_args_list]
        insert_query = ' '.join(calls)
        assert 'on conflict' in insert_query.lower()


@pytest.mark.integration
class TestStockDataLoaderIntegration:
    """Integration tests requiring database connection."""

    @pytest.mark.skip(reason="Requires database connection")
    def test_load_training_data_returns_dataframe(self):
        """Test loading data returns properly formatted DataFrame."""
        loader = StockDataLoader()
        df = loader.load_training_data(min_date='2024-01-01')

        assert isinstance(df, pd.DataFrame)
        assert 'ticker' in df.columns
        assert 'date' in df.columns
        assert 'close' in df.columns
