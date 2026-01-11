"""Feature engineering for volatility prediction."""
import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_volatility(df: pd.DataFrame) -> pd.Series:
    """Calculate intraday volatility as (high - low) / close * 100."""
    return ((df['high'] - df['low']) / df['close'] * 100).round(4)


def classify_volatility(volatility: pd.Series, thresholds: Dict[str, float] = None) -> pd.Series:
    """
    Classify volatility into low/medium/high categories.

    Default thresholds:
    - low: < 2%
    - medium: 2% - 5%
    - high: > 5%
    """
    if thresholds is None:
        thresholds = {'medium': 2.0, 'high': 5.0}

    return pd.cut(
        volatility,
        bins=[0, thresholds['medium'], thresholds['high'], float('inf')],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(4)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.round(4)


def calculate_bollinger_width(prices: pd.Series, period: int = 20, num_std: float = 2) -> pd.Series:
    """Calculate Bollinger Bands width as percentage."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    width = ((upper_band - lower_band) / sma * 100).round(4)

    return width


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day of week and month features."""
    df = df.copy()
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    return df


def add_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Add lagged features for specified columns."""
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby('ticker')[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, column: str, windows: List[int],
                        agg_funcs: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """Add rolling statistics for a column."""
    df = df.copy()
    for window in windows:
        for func in agg_funcs:
            col_name = f'{column}_rolling_{window}_{func}'
            df[col_name] = df.groupby('ticker')[column].transform(
                lambda x: x.rolling(window=window, min_periods=1).agg(func)
            ).round(4)
    return df


def engineer_features(df: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    Args:
        df: DataFrame with columns: ticker, date, open, high, low, close, volume, sentiment, topic
        for_prediction: If True, don't create target variable (for inference)

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Calculate volatility (this becomes our target if not for_prediction)
    df['volatility'] = df.groupby('ticker', group_keys=False).apply(
        lambda x: calculate_volatility(x)
    ).reset_index(drop=True)

    # Calculate returns
    df['return_1d'] = df.groupby('ticker')['close'].pct_change() * 100
    df['return_3d'] = df.groupby('ticker')['close'].pct_change(periods=3) * 100
    df['return_5d'] = df.groupby('ticker')['close'].pct_change(periods=5) * 100

    # Technical indicators
    df['rsi'] = df.groupby('ticker', group_keys=False).apply(
        lambda x: calculate_rsi(x['close'])
    ).reset_index(drop=True)

    df['atr'] = df.groupby('ticker', group_keys=False).apply(
        lambda x: calculate_atr(x)
    ).reset_index(drop=True)

    df['bollinger_width'] = df.groupby('ticker', group_keys=False).apply(
        lambda x: calculate_bollinger_width(x['close'])
    ).reset_index(drop=True)

    # Volume features
    df['volume_ma_20'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    df['volume_ratio'] = (df['volume'] / df['volume_ma_20']).round(4)

    # Rolling volatility statistics
    df = add_rolling_features(df, 'volatility', windows=[3, 5, 10, 20], agg_funcs=['mean', 'std'])

    # Rolling return statistics
    df = add_rolling_features(df, 'return_1d', windows=[5, 10], agg_funcs=['mean', 'std'])

    # Lag features for volatility
    df = add_lag_features(df, ['volatility', 'volume_ratio'], lags=[1, 2, 3])

    # Temporal features
    df = add_temporal_features(df)

    # Encode categorical variables
    if 'sentiment' in df.columns:
        df['sentiment_encoded'] = df['sentiment'].map({
            'positive': 1, 'neutral': 0, 'negative': -1
        }).fillna(0)

    if 'topic' in df.columns:
        # One-hot encode topic
        topic_dummies = pd.get_dummies(df['topic'], prefix='topic')
        df = pd.concat([df, topic_dummies], axis=1)

    # Create target variable (next day's volatility) if not for prediction
    if not for_prediction:
        df['target_volatility'] = df.groupby('ticker')['volatility'].shift(-1)
        df['target_volatility_class'] = df.groupby('ticker', group_keys=False).apply(
            lambda x: classify_volatility(x['target_volatility'])
        ).reset_index(drop=True)

    return df


def get_feature_columns() -> List[str]:
    """Get list of feature columns to use for modeling."""
    base_features = [
        'close', 'volume', 'return_1d', 'return_3d', 'return_5d',
        'rsi', 'atr', 'bollinger_width', 'volume_ratio',
        'day_of_week', 'month', 'sentiment_encoded'
    ]

    # Volatility rolling features
    volatility_rolling = [
        f'volatility_rolling_{w}_{agg}'
        for w in [3, 5, 10, 20]
        for agg in ['mean', 'std']
    ]

    # Return rolling features
    return_rolling = [
        f'return_1d_rolling_{w}_{agg}'
        for w in [5, 10]
        for agg in ['mean', 'std']
    ]

    # Lag features
    lag_features = [
        f'{col}_lag_{lag}'
        for col in ['volatility', 'volume_ratio']
        for lag in [1, 2, 3]
    ]

    # Topic features (will be added dynamically if present)
    topic_features = [
        'topic_earnings', 'topic_macro',
        'topic_company-specific', 'topic_speculation'
    ]

    return base_features + volatility_rolling + return_rolling + lag_features + topic_features


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                             val_size: float = 0.1) -> tuple:
    """
    Time-series aware train/validation/test split.

    Args:
        df: Engineered features DataFrame
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)
    """
    # Remove rows with NaN in target
    df = df.dropna(subset=['target_volatility_class'])

    # Encode target labels to integers for XGBoost compatibility
    label_map = {'low': 0, 'medium': 1, 'high': 2}
    df['target_volatility_class'] = df['target_volatility_class'].map(label_map)

    # Get feature columns that exist in the dataframe
    all_features = get_feature_columns()
    feature_cols = [col for col in all_features if col in df.columns]

    # Sort by date for time-series split
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate split indices
    n = len(df)
    test_idx = int(n * (1 - test_size))
    train_idx = int(test_idx * (1 - val_size))

    # Split data
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:test_idx]
    test_df = df.iloc[test_idx:]

    # Remove rows with NaN in features
    train_df = train_df.dropna(subset=feature_cols)
    val_df = val_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)

    # Prepare X and y
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df['target_volatility_class']
    y_val = val_df['target_volatility_class']
    y_test = test_df['target_volatility_class']

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
