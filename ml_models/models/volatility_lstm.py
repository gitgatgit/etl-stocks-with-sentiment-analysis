import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pytorch


class StockVolatilityDataset(Dataset):
    """Dataset for stock volatility prediction"""

    def __init__(self, data, seq_length=20):
        """
        Initialize dataset

        Args:
            data: Array of returns
            seq_length: Length of input sequence
        """
        self.seq_length = seq_length
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Get sequence of returns
        x = self.data[idx:idx + self.seq_length]
        # Target is the squared return (proxy for volatility)
        y = self.data[idx + self.seq_length] ** 2

        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMVolatilityModel(nn.Module):
    """LSTM model for volatility forecasting"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize LSTM model

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMVolatilityModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive volatility
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        volatility = self.fc(last_output)

        return volatility


class LSTMVolatilityTrainer:
    """Trainer for LSTM volatility model"""

    def __init__(self, seq_length=20, hidden_size=64, num_layers=2,
                 dropout=0.2, learning_rate=0.001, batch_size=32):
        """
        Initialize trainer

        Args:
            seq_length: Length of input sequence
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, df):
        """Prepare return data"""
        # Calculate log returns
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        # Convert to percentage returns
        returns = returns.values * 100
        return returns

    def train(self, df, ticker, epochs=50, validation_split=0.2):
        """
        Train LSTM model

        Args:
            df: DataFrame with stock prices
            ticker: Stock ticker symbol
            epochs: Number of training epochs
            validation_split: Fraction of data for validation

        Returns:
            Training history
        """
        # Prepare data
        returns = self.prepare_data(df)

        # Normalize data
        returns_scaled = self.scaler.fit_transform(returns.reshape(-1, 1)).flatten()

        # Split into train and validation
        split_idx = int(len(returns_scaled) * (1 - validation_split))
        train_data = returns_scaled[:split_idx]
        val_data = returns_scaled[split_idx:]

        # Create datasets
        train_dataset = StockVolatilityDataset(train_data, self.seq_length)
        val_dataset = StockVolatilityDataset(val_data, self.seq_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        self.model = LSTMVolatilityModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.unsqueeze(-1).to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.unsqueeze(-1).to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        return history

    def forecast(self, df, horizon=1):
        """
        Forecast volatility

        Args:
            df: DataFrame with recent stock prices
            horizon: Number of periods ahead to forecast

        Returns:
            Array of forecasted volatility
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")

        self.model.eval()

        returns = self.prepare_data(df)
        returns_scaled = self.scaler.transform(returns.reshape(-1, 1)).flatten()

        # Take last seq_length returns
        last_sequence = returns_scaled[-self.seq_length:]

        forecasts = []
        current_seq = last_sequence.copy()

        with torch.no_grad():
            for _ in range(horizon):
                X = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
                volatility = self.model(X).cpu().numpy()[0, 0]
                forecasts.append(np.sqrt(volatility))  # Convert variance to volatility

                # For multi-step, we'd need to simulate next return
                # For simplicity, using last volatility forecast
                if horizon > 1:
                    current_seq = np.roll(current_seq, -1)
                    current_seq[-1] = 0  # Placeholder

        return np.array(forecasts)

    def evaluate(self, df_test, ticker):
        """
        Evaluate model performance

        Args:
            df_test: Test DataFrame
            ticker: Stock ticker

        Returns:
            Dictionary of evaluation metrics
        """
        returns = self.prepare_data(df_test)
        returns_scaled = self.scaler.transform(returns.reshape(-1, 1)).flatten()

        # Calculate realized volatility
        realized_vol = pd.Series(returns).rolling(window=5).std().values

        # Generate forecasts
        forecasts = []
        for i in range(self.seq_length, len(returns_scaled) - 5):
            seq = returns_scaled[i-self.seq_length:i]
            X = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                vol = self.model(X).cpu().numpy()[0, 0]
                forecasts.append(np.sqrt(vol))

        forecasts = np.array(forecasts)
        actual = realized_vol[self.seq_length+5:]

        # Remove NaN values
        mask = ~np.isnan(actual)
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
            'model_type': 'LSTM',
            'ticker': ticker
        }
