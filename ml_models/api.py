from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import psycopg2
import mlflow
import mlflow.pyfunc
import os
from datetime import datetime, timedelta
import numpy as np

app = FastAPI(
    title="Volatility Forecasting API",
    description="API for stock volatility predictions using GARCH and LSTM models",
    version="1.0.0"
)

# MLflow setup
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASS', 'airflow')
    )


def load_stock_data(ticker, days_back=90):
    """Load recent stock price data"""
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


def get_latest_model(model_name):
    """Get the latest production or staging model version"""
    try:
        client = mlflow.MlflowClient()

        # Try to get production version first
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        except:
            pass

        # If no production version, get latest version
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            latest_version = versions[0].version
            return mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

        return None
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None


class PredictionRequest(BaseModel):
    ticker: str
    model_type: str = "GARCH"  # GARCH or LSTM
    horizon: int = 1


class PredictionResponse(BaseModel):
    ticker: str
    model_type: str
    horizon: int
    forecasts: List[float]
    forecast_dates: List[str]
    timestamp: str


class ModelInfo(BaseModel):
    model_name: str
    ticker: str
    version: Optional[str]
    stage: Optional[str]
    metrics: Optional[dict]


@app.get("/")
def root():
    """API root endpoint"""
    return {
        "service": "Volatility Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "models": "/models",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        conn = get_db_connection()
        conn.close()

        # Check MLflow connection
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()

        return {
            "status": "healthy",
            "database": "connected",
            "mlflow": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    """List all available models"""
    try:
        client = mlflow.MlflowClient()
        models_info = []

        # Get all registered models
        for model in client.search_registered_models():
            model_name = model.name

            # Get ticker from model name (format: garch_volatility_aapl)
            parts = model_name.split('_')
            if len(parts) >= 3:
                ticker = parts[-1].upper()
                model_type = parts[0].upper()

                # Get latest version
                versions = client.get_latest_versions(model_name)
                if versions:
                    version = versions[0].version
                    stage = versions[0].current_stage

                    # Get metrics if available
                    try:
                        run = client.get_run(versions[0].run_id)
                        metrics = run.data.metrics
                    except:
                        metrics = None

                    models_info.append(ModelInfo(
                        model_name=model_name,
                        ticker=ticker,
                        version=version,
                        stage=stage,
                        metrics=metrics
                    ))

        return models_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_volatility(request: PredictionRequest):
    """
    Predict volatility for a stock

    Args:
        request: PredictionRequest with ticker, model_type, and horizon

    Returns:
        PredictionResponse with forecasts
    """
    try:
        ticker = request.ticker.upper()
        model_type = request.model_type.upper()
        horizon = request.horizon

        if model_type not in ['GARCH', 'LSTM']:
            raise HTTPException(status_code=400, detail="model_type must be GARCH or LSTM")

        if horizon < 1 or horizon > 30:
            raise HTTPException(status_code=400, detail="horizon must be between 1 and 30")

        # Load stock data
        df = load_stock_data(ticker, days_back=365)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")

        # Load model
        model_name = f"{model_type.lower()}_volatility_{ticker.lower()}"
        model = get_latest_model(model_name)

        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained {model_type} model found for {ticker}. Train the model first."
            )

        # Generate predictions based on model type
        if model_type == 'GARCH':
            # For GARCH, we need to use the saved model object
            # This is a simplified version - in production, you'd reconstruct the GARCH model
            forecasts = [2.5] * horizon  # Placeholder - implement actual GARCH prediction

        else:  # LSTM
            # For LSTM, use MLflow model
            # This is simplified - you'd need to properly format input data
            forecasts = [2.0] * horizon  # Placeholder - implement actual LSTM prediction

        # Generate forecast dates (business days)
        last_date = pd.to_datetime(df['date'].iloc[-1])
        forecast_dates = []
        current_date = last_date

        for _ in range(horizon):
            current_date += timedelta(days=1)
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            forecast_dates.append(current_date.strftime('%Y-%m-%d'))

        # Store predictions in database
        store_predictions(ticker, model_type, forecasts, forecast_dates)

        return PredictionResponse(
            ticker=ticker,
            model_type=model_type,
            horizon=horizon,
            forecasts=forecasts,
            forecast_dates=forecast_dates,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def store_predictions(ticker, model_type, forecasts, forecast_dates):
    """Store predictions in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for forecast, forecast_date in zip(forecasts, forecast_dates):
            cursor.execute("""
                INSERT INTO ml.volatility_predictions
                (ticker, model_type, forecast_date, predicted_volatility, prediction_timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ticker, model_type, forecast_date)
                DO UPDATE SET
                    predicted_volatility = EXCLUDED.predicted_volatility,
                    prediction_timestamp = EXCLUDED.prediction_timestamp
            """, (ticker, model_type, forecast_date, forecast, datetime.now()))

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error storing predictions: {str(e)}")


@app.get("/predictions/{ticker}")
def get_predictions(ticker: str, days: int = 7):
    """Get recent predictions for a ticker"""
    try:
        conn = get_db_connection()

        query = """
        SELECT ticker, model_type, forecast_date, predicted_volatility, prediction_timestamp
        FROM ml.volatility_predictions
        WHERE ticker = %s
        AND forecast_date >= CURRENT_DATE
        AND forecast_date <= CURRENT_DATE + INTERVAL '%s days'
        ORDER BY model_type, forecast_date
        """

        df = pd.read_sql_query(query, conn, params=(ticker.upper(), days))
        conn.close()

        if df.empty:
            return {"ticker": ticker.upper(), "predictions": []}

        predictions = df.to_dict('records')

        return {
            "ticker": ticker.upper(),
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
