import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pyfunc
import pickle


class GARCHVolatilityModel:
    """GARCH(1,1) model for volatility forecasting"""

    def __init__(self, p=1, q=1):
        """
        Initialize GARCH model

        Args:
            p: Order of GARCH term
            q: Order of ARCH term
        """
        self.p = p
        self.q = q
        self.model = None
        self.model_fit = None

    def prepare_data(self, df):
        """Prepare return data for GARCH modeling"""
        # Calculate log returns
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        # Convert to percentage returns
        returns = returns * 100
        return returns

    def train(self, df, ticker):
        """
        Train GARCH model

        Args:
            df: DataFrame with stock prices
            ticker: Stock ticker symbol
        """
        returns = self.prepare_data(df)

        # Build GARCH model
        self.model = arch_model(
            returns,
            vol='Garch',
            p=self.p,
            q=self.q,
            dist='normal'
        )

        # Fit the model
        self.model_fit = self.model.fit(disp='off')

        return self.model_fit

    def forecast(self, horizon=1):
        """
        Forecast volatility

        Args:
            horizon: Number of periods ahead to forecast

        Returns:
            DataFrame with forecasted volatility
        """
        if self.model_fit is None:
            raise ValueError("Model must be trained before forecasting")

        # Generate forecast
        forecast = self.model_fit.forecast(horizon=horizon)

        # Extract volatility forecast (annualized)
        volatility_forecast = np.sqrt(forecast.variance.values[-1, :])

        return volatility_forecast

    def evaluate(self, df_test, ticker):
        """
        Evaluate model performance on test set

        Args:
            df_test: Test DataFrame
            ticker: Stock ticker

        Returns:
            Dictionary of evaluation metrics
        """
        returns = self.prepare_data(df_test)

        # Calculate realized volatility (rolling std of returns)
        realized_vol = returns.rolling(window=5).std()

        # Generate forecasts for each point
        forecasts = []
        for i in range(len(returns) - 5):
            train_data = returns.iloc[:len(returns)-5+i]
            temp_model = arch_model(train_data, vol='Garch', p=self.p, q=self.q)
            temp_fit = temp_model.fit(disp='off')
            forecast = temp_fit.forecast(horizon=1)
            forecasts.append(np.sqrt(forecast.variance.values[-1, 0]))

        forecasts = np.array(forecasts)
        actual = realized_vol.iloc[5:].values

        # Remove NaN values
        mask = ~np.isnan(actual) & ~np.isnan(forecasts)
        actual = actual[mask]
        forecasts = forecasts[mask]

        # Calculate metrics
        mse = mean_squared_error(actual, forecasts)
        mae = mean_absolute_error(actual, forecasts)
        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'model_type': 'GARCH',
            'ticker': ticker
        }


class GARCHModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for GARCH model"""

    def load_context(self, context):
        """Load the GARCH model"""
        with open(context.artifacts["model_path"], 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        """
        Make predictions

        Args:
            model_input: DataFrame with 'horizon' column

        Returns:
            Array of volatility forecasts
        """
        horizon = int(model_input['horizon'].iloc[0]) if 'horizon' in model_input else 1
        return self.model.forecast(horizon=horizon)
