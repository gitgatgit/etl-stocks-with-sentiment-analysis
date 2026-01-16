"""Pytest fixtures and configuration."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = {
        'ticker': ['AAPL'] * 100,
        'date': dates,
        'open': np.random.uniform(150, 180, 100),
        'high': np.random.uniform(155, 185, 100),
        'low': np.random.uniform(145, 175, 100),
        'close': np.random.uniform(150, 180, 100),
        'volume': np.random.randint(1000000, 10000000, 100),
    }

    df = pd.DataFrame(data)
    # Ensure high >= low
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_predictions():
    """Create sample ML predictions for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')

    return pd.DataFrame({
        'ticker': ['AAPL'] * 10,
        'date': dates,
        'predicted_volatility_class': ['low', 'medium', 'high'] * 3 + ['low'],
        'predicted_volatility': np.random.uniform(0, 10, 10),
        'confidence': np.random.uniform(0.5, 1.0, 10),
        'model_version': ['v1.0'] * 10
    })


@pytest.fixture
def mock_db_connection(mocker):
    """Mock database connection for testing."""
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor
