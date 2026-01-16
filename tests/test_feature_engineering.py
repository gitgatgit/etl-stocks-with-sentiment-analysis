"""Tests for feature engineering module."""
import pytest
import pandas as pd
import numpy as np
from ml.feature_engineering import (
    calculate_volatility,
    classify_volatility,
    calculate_rsi,
    calculate_atr
)


@pytest.mark.unit
class TestVolatilityCalculations:
    """Test core volatility calculation logic."""

    def test_calculate_volatility(self, sample_stock_data):
        """Test volatility calculation returns valid values."""
        volatility = calculate_volatility(sample_stock_data)

        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(sample_stock_data)
        assert (volatility >= 0).all(), "Volatility should be non-negative"

    def test_calculate_volatility_zero_close(self):
        """Test volatility calculation handles zero close price."""
        df = pd.DataFrame({
            'high': [10],
            'low': [5],
            'close': [0]
        })

        volatility = calculate_volatility(df)
        assert volatility.isna().all() or (volatility == np.inf).all()

    def test_classify_volatility_thresholds(self):
        """Test volatility classification logic with known values."""
        volatility = pd.Series([1.0, 3.0, 7.0])  # low, medium, high
        classification = classify_volatility(volatility)

        assert classification[0] == 'low'
        assert classification[1] == 'medium'
        assert classification[2] == 'high'

    def test_classify_volatility_custom_thresholds(self):
        """Test volatility classification with custom thresholds."""
        volatility = pd.Series([1, 3, 7, 10])
        thresholds = {'medium': 2.5, 'high': 6.0}

        classification = classify_volatility(volatility, thresholds)

        assert classification[0] == 'low'
        assert classification[1] == 'medium'
        assert classification[2] == 'high'
        assert classification[3] == 'high'


@pytest.mark.unit
class TestTechnicalIndicators:
    """Test technical indicators return expected value ranges."""

    def test_calculate_rsi_range(self, sample_stock_data):
        """Test RSI values are in valid 0-100 range."""
        rsi = calculate_rsi(sample_stock_data['close'], period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_calculate_atr_positive(self, sample_stock_data):
        """Test ATR values are non-negative."""
        atr = calculate_atr(sample_stock_data, period=14)

        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
