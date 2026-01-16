#!/usr/bin/env python3
"""Make predictions using trained volatility model."""
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ml.data_loader import StockDataLoader
from ml.feature_engineering import engineer_features
from ml.mlflow_utils import MLflowTracker


class VolatilityPredictor:
    """Make volatility predictions using trained model."""

    # Map numeric classes to human-readable labels
    CLASS_LABELS = {0: 'low', 1: 'medium', 2: 'high'}

    def __init__(self, model_path: str = 'ml/models/latest.pkl',
                 metadata_path: str = 'ml/models/latest_metadata.json',
                 db_host: str = 'localhost'):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to pickled model file
            metadata_path: Path to model metadata JSON
            db_host: Database host for loading data
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.db_host = db_host
        self.model = None
        self.metadata = None
        self.feature_cols = None

        self.load_model()

    def load_model(self):
        """Load model and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Loaded model from: {self.model_path}")

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_cols = self.metadata['feature_columns']
            print(f"✓ Loaded metadata from: {self.metadata_path}")
            print(f"  Model type: {self.metadata['model_type']}")
            print(f"  Trained at: {self.metadata['trained_at']}")
        else:
            print(f"Warning: Metadata file not found: {self.metadata_path}")

    def predict(self, df: pd.DataFrame, return_proba: bool = False) -> pd.DataFrame:
        """
        Make predictions on data.

        Args:
            df: DataFrame with features
            return_proba: If True, return class probabilities

        Returns:
            DataFrame with predictions
        """
        # Ensure we have all required features
        missing_cols = set(self.feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        X = df[self.feature_cols]

        # Make predictions
        predictions = self.model.predict(X)

        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            # Confidence is the max probability
            confidence = probabilities.max(axis=1)
        else:
            confidence = np.ones(len(predictions))

        # Create results DataFrame with human-readable labels
        results = pd.DataFrame({
            'ticker': df['ticker'].values,
            'date': df['date'].values,
            'predicted_volatility_class': [self.CLASS_LABELS[p] for p in predictions],
            'confidence': confidence
        })

        if return_proba and hasattr(self.model, 'predict_proba'):
            # Add probability for each class
            results['prob_low'] = probabilities[:, 0]
            results['prob_medium'] = probabilities[:, 1]
            results['prob_high'] = probabilities[:, 2]

        return results

    def predict_next_day(self, tickers: list = None, save_to_db: bool = False) -> pd.DataFrame:
        """
        Predict volatility for the next trading day.

        Args:
            tickers: List of tickers to predict (default: all available)
            save_to_db: If True, save predictions to database

        Returns:
            DataFrame with predictions
        """
        print("\nPredicting next-day volatility...")

        # Load recent data for feature engineering
        loader = StockDataLoader(host=self.db_host)
        df = loader.load_training_data()

        if tickers:
            df = df[df['ticker'].isin(tickers)]

        print(f"✓ Loaded {len(df)} historical records")

        # Engineer features (for prediction mode)
        df_features = engineer_features(df, for_prediction=True)

        # Get the most recent date for each ticker
        latest_data = df_features.groupby('ticker').tail(1).reset_index(drop=True)

        print(f"✓ Prepared features for {len(latest_data)} tickers")

        # Make predictions
        predictions = self.predict(latest_data, return_proba=True)

        # Add next trading day date (assuming next day, adjust for weekends)
        latest_dates = pd.to_datetime(latest_data['date'])
        next_dates = latest_dates + timedelta(days=1)

        # Skip weekends (simple approach)
        next_dates = next_dates.apply(lambda x: x if x.weekday() < 5 else x + timedelta(days=(7 - x.weekday())))

        predictions['date'] = next_dates.values
        predictions['model_version'] = self.metadata.get('model_name', 'v1.0') if self.metadata else 'v1.0'

        # Save to database if requested
        if save_to_db:
            print("\nSaving predictions to database...")
            loader.save_predictions(predictions)

        return predictions


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(description='Make volatility predictions')
    parser.add_argument('--model', type=str, default='ml/models/latest.pkl',
                       help='Path to trained model file')
    parser.add_argument('--metadata', type=str, default='ml/models/latest_metadata.json',
                       help='Path to model metadata file')
    parser.add_argument('--tickers', type=str, nargs='+', default=None,
                       help='Tickers to predict (default: all)')
    parser.add_argument('--save-db', action='store_true',
                       help='Save predictions to database')
    parser.add_argument('--output', type=str, default=None,
                       help='Save predictions to CSV file')
    parser.add_argument('--db-host', type=str, default='localhost',
                       help='Database host')

    args = parser.parse_args()

    print("=" * 60)
    print("VOLATILITY PREDICTION")
    print("=" * 60)

    # Initialize predictor
    predictor = VolatilityPredictor(
        model_path=args.model,
        metadata_path=args.metadata,
        db_host=args.db_host
    )

    # Make predictions
    predictions = predictor.predict_next_day(
        tickers=args.tickers,
        save_to_db=args.save_db
    )

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    print(predictions.to_string(index=False))

    # Show summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(predictions['predicted_volatility_class'].value_counts())
    print(f"\nAverage confidence: {predictions['confidence'].mean():.3f}")

    # Save to CSV if requested
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\n✓ Predictions saved to: {args.output}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
