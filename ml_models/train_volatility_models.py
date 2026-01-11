import os
import sys
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import pickle
import torch

# Add models to path
sys.path.append(os.path.dirname(__file__))

from models.volatility_garch import GARCHVolatilityModel, GARCHModelWrapper
from models.volatility_lstm import LSTMVolatilityTrainer


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASS', 'airflow')
    )


def load_stock_data(ticker, days_back=365):
    """
    Load stock price data from database

    Args:
        ticker: Stock ticker symbol
        days_back: Number of days of historical data

    Returns:
        DataFrame with stock prices
    """
    conn = get_db_connection()

    query = """
    SELECT date, open, high, low, close, volume
    FROM raw.stock_prices
    WHERE ticker = %s
    AND date >= CURRENT_DATE - INTERVAL '%s days'
    ORDER BY date ASC
    """

    df = pd.read_sql_query(query, conn, params=(ticker, days_back))
    conn.close()

    return df


def train_garch_model(ticker, experiment_name="volatility-forecasting"):
    """
    Train GARCH model for a ticker

    Args:
        ticker: Stock ticker symbol
        experiment_name: MLflow experiment name

    Returns:
        Model metrics and run ID
    """
    print(f"Training GARCH model for {ticker}")

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Load data
    df = load_stock_data(ticker, days_back=365)

    if len(df) < 100:
        print(f"Not enough data for {ticker}. Skipping.")
        return None

    # Split train/test (80/20)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    with mlflow.start_run(run_name=f"GARCH_{ticker}"):
        # Log parameters
        mlflow.log_param("model_type", "GARCH")
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("p", 1)
        mlflow.log_param("q", 1)
        mlflow.log_param("train_samples", len(df_train))
        mlflow.log_param("test_samples", len(df_test))

        # Train model
        model = GARCHVolatilityModel(p=1, q=1)
        model.train(df_train, ticker)

        # Evaluate on test set
        metrics = model.evaluate(df_test, ticker)

        # Log metrics
        mlflow.log_metric("test_mse", metrics['mse'])
        mlflow.log_metric("test_mae", metrics['mae'])
        mlflow.log_metric("test_rmse", metrics['rmse'])

        # Generate forecast
        forecast_1d = model.forecast(horizon=1)
        forecast_5d = model.forecast(horizon=5)

        mlflow.log_metric("forecast_1d", float(forecast_1d[0]))
        mlflow.log_metric("forecast_5d_avg", float(forecast_5d.mean()))

        # Save model
        model_path = f"/tmp/garch_{ticker}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Log model with MLflow
        artifacts = {"model_path": model_path}
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=GARCHModelWrapper(),
            artifacts=artifacts,
            registered_model_name=f"garch_volatility_{ticker.lower()}"
        )

        run_id = mlflow.active_run().info.run_id

        print(f"GARCH model for {ticker} trained. Run ID: {run_id}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"1-day forecast: {forecast_1d[0]:.4f}")

        return {
            'run_id': run_id,
            'metrics': metrics,
            'ticker': ticker,
            'model_type': 'GARCH'
        }


def train_lstm_model(ticker, experiment_name="volatility-forecasting", epochs=50):
    """
    Train LSTM model for a ticker

    Args:
        ticker: Stock ticker symbol
        experiment_name: MLflow experiment name
        epochs: Number of training epochs

    Returns:
        Model metrics and run ID
    """
    print(f"Training LSTM model for {ticker}")

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Load data
    df = load_stock_data(ticker, days_back=730)  # 2 years for LSTM

    if len(df) < 200:
        print(f"Not enough data for {ticker}. Skipping.")
        return None

    # Split train/test (80/20)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    with mlflow.start_run(run_name=f"LSTM_{ticker}"):
        # Log parameters
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("seq_length", 20)
        mlflow.log_param("hidden_size", 64)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_samples", len(df_train))
        mlflow.log_param("test_samples", len(df_test))

        # Train model
        trainer = LSTMVolatilityTrainer(
            seq_length=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            batch_size=32
        )

        history = trainer.train(df_train, ticker, epochs=epochs, validation_split=0.2)

        # Log training history
        for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Evaluate on test set
        metrics = trainer.evaluate(df_test, ticker)

        # Log metrics
        mlflow.log_metric("test_mse", metrics['mse'])
        mlflow.log_metric("test_mae", metrics['mae'])
        mlflow.log_metric("test_rmse", metrics['rmse'])

        # Generate forecast
        forecast_1d = trainer.forecast(df, horizon=1)
        forecast_5d = trainer.forecast(df, horizon=5)

        mlflow.log_metric("forecast_1d", float(forecast_1d[0]))
        mlflow.log_metric("forecast_5d_avg", float(forecast_5d.mean()))

        # Log PyTorch model
        mlflow.pytorch.log_model(
            trainer.model,
            "model",
            registered_model_name=f"lstm_volatility_{ticker.lower()}"
        )

        # Save scaler
        scaler_path = f"/tmp/scaler_{ticker}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(trainer.scaler, f)

        mlflow.log_artifact(scaler_path, "scaler")

        run_id = mlflow.active_run().info.run_id

        print(f"LSTM model for {ticker} trained. Run ID: {run_id}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"1-day forecast: {forecast_1d[0]:.4f}")

        return {
            'run_id': run_id,
            'metrics': metrics,
            'ticker': ticker,
            'model_type': 'LSTM'
        }


def train_all_models(tickers=None, model_types=['GARCH', 'LSTM']):
    """
    Train all models for all tickers

    Args:
        tickers: List of tickers (default: all tickers in DB)
        model_types: List of model types to train

    Returns:
        List of training results
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))

    if tickers is None:
        # Get all tickers from database
        conn = get_db_connection()
        query = "SELECT DISTINCT ticker FROM raw.stock_prices ORDER BY ticker"
        df = pd.read_sql_query(query, conn)
        conn.close()
        tickers = df['ticker'].tolist()

    print(f"Training models for tickers: {tickers}")
    print(f"Model types: {model_types}")

    results = []

    for ticker in tickers:
        try:
            if 'GARCH' in model_types:
                garch_result = train_garch_model(ticker)
                if garch_result:
                    results.append(garch_result)

            if 'LSTM' in model_types:
                lstm_result = train_lstm_model(ticker, epochs=50)
                if lstm_result:
                    results.append(lstm_result)

        except Exception as e:
            print(f"Error training models for {ticker}: {str(e)}")
            continue

    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)

    for result in results:
        print(f"{result['model_type']} - {result['ticker']}: RMSE = {result['metrics']['rmse']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train volatility forecasting models')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to train')
    parser.add_argument('--models', nargs='+', default=['GARCH', 'LSTM'],
                       help='Model types to train (GARCH, LSTM)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for LSTM')

    args = parser.parse_args()

    results = train_all_models(tickers=args.tickers, model_types=args.models)

    print(f"\nTrained {len(results)} models successfully!")
