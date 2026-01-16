#!/usr/bin/env python3
"""Train volatility prediction model."""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score
)
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, will use RandomForest only")

from ml.data_loader import StockDataLoader
from ml.feature_engineering import engineer_features, prepare_train_test_split
from ml.mlflow_utils import MLflowTracker, create_model_signature


class VolatilityModelTrainer:
    """Train and evaluate volatility prediction models."""

    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize trainer.

        Args:
            model_type: 'xgboost' or 'random_forest'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
        self.metrics = {}

    def create_model(self):
        """Create model based on type."""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=3,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for XGBoost early stopping)
            y_val: Validation labels (optional, for XGBoost early stopping)
        """
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Class distribution:\n{y_train.value_counts()}\n")

        # Compute sample weights to handle class imbalance
        sample_weights = compute_sample_weight('balanced', y_train)
        print("✓ Applied balanced class weights to handle imbalance")

        self.create_model()

        if self.model_type == 'xgboost' and X_val is not None and XGBOOST_AVAILABLE:
            # Use early stopping for XGBoost
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)

        print("✓ Training completed")

    def evaluate(self, X_test, y_test, dataset_name: str = 'Test') -> dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of metrics
        """
        print(f"\n{dataset_name} Set Evaluation:")
        print("=" * 50)

        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print(f"Recall (Weighted): {recall_weighted:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop 15 Important Features:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            for idx, row in feature_importance.head(15).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'recall_macro': float(recall_macro),
            'recall_weighted': float(recall_weighted),
            'confusion_matrix': cm.tolist(),
            'dataset': dataset_name,
            'n_samples': len(y_test)
        }

        return metrics

    def save_model(self, output_dir: str = 'ml/models', model_name: str = None):
        """
        Save trained model and metadata.

        Args:
            output_dir: Directory to save model
            model_name: Custom model name (default: auto-generated)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"volatility_{self.model_type}_{timestamp}"

        # Save model
        model_file = output_path / f"{model_name}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n✓ Model saved to: {model_file}")

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'model_name': model_name,
            'trained_at': datetime.now().isoformat(),
            'feature_columns': self.feature_cols,
            'metrics': self.metrics,
            'random_state': self.random_state
        }

        metadata_file = output_path / f"{model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_file}")

        # Create a 'latest' symlink for easy access
        latest_link = output_path / 'latest.pkl'
        latest_metadata_link = output_path / 'latest_metadata.json'

        if latest_link.exists():
            latest_link.unlink()
        if latest_metadata_link.exists():
            latest_metadata_link.unlink()

        latest_link.symlink_to(model_file.name)
        latest_metadata_link.symlink_to(metadata_file.name)

        print(f"✓ Latest model linked")

        return model_file, metadata_file


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train volatility prediction model')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest'],
                       help='Model type to train')
    parser.add_argument('--min-date', type=str, default=None,
                       help='Minimum date for training data (YYYY-MM-DD)')
    parser.add_argument('--max-date', type=str, default=None,
                       help='Maximum date for training data (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='ml/models',
                       help='Directory to save trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for test set')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Proportion of training data for validation')
    parser.add_argument('--db-host', type=str, default='localhost',
                       help='Database host')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    parser.add_argument('--mlflow-experiment', type=str, default='volatility-prediction',
                       help='MLflow experiment name')

    args = parser.parse_args()

    print("=" * 60)
    print("VOLATILITY PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Initialize MLflow tracking
    mlflow_tracker = None
    if not args.no_mlflow:
        try:
            mlflow_tracker = MLflowTracker(experiment_name=args.mlflow_experiment)
            print("\n✓ MLflow tracking enabled")
        except Exception as e:
            print(f"\n⚠ MLflow tracking disabled: {e}")
            mlflow_tracker = None

    # Load data
    print("\n1. Loading data from database...")
    loader = StockDataLoader(host=args.db_host)

    stats = loader.get_data_statistics()
    print(f"\nData statistics:")
    print(f"  Total stock prices: {stats['stock_prices']['count']}")
    print(f"  Date range: {stats['stock_prices']['min_date']} to {stats['stock_prices']['max_date']}")
    print(f"  Analytics records: {stats['analytics_records']}")
    print(f"  Tickers: {list(stats['by_ticker'].keys())}")

    df = loader.load_training_data(min_date=args.min_date, max_date=args.max_date)
    print(f"\n✓ Loaded {len(df)} records")

    if len(df) < 100:
        print("\n⚠ Warning: Very few records available for training!")
        print("Consider running the ETL pipeline to collect more data first.")
        return

    # Feature engineering
    print("\n2. Engineering features...")
    df_features = engineer_features(df, for_prediction=False)
    print(f"✓ Created {len(df_features.columns)} feature columns")

    # Prepare train/val/test split
    print("\n3. Preparing train/validation/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_train_test_split(
        df_features,
        test_size=args.test_size,
        val_size=args.val_size
    )

    print(f"  Train set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Train model
    print("\n4. Training model...")
    trainer = VolatilityModelTrainer(model_type=args.model)
    trainer.feature_cols = feature_cols

    # Start MLflow run
    if mlflow_tracker:
        run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_tracker.start_run(run_name=run_name, tags={
            'model_type': args.model,
            'framework': 'xgboost' if args.model == 'xgboost' else 'sklearn'
        })

        # Log parameters
        params = {
            'model_type': args.model,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'n_features': len(feature_cols),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'random_state': trainer.random_state
        }

        # Add model-specific hyperparameters
        if args.model == 'xgboost':
            params.update({
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'multi:softmax'
            })
        else:
            params.update({
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 4
            })

        mlflow_tracker.log_params(params)

    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    print("\n5. Evaluating model...")
    val_metrics = trainer.evaluate(X_val, y_val, dataset_name='Validation')
    test_metrics = trainer.evaluate(X_test, y_test, dataset_name='Test')

    trainer.metrics = {
        'validation': val_metrics,
        'test': test_metrics
    }

    # Log metrics to MLflow
    if mlflow_tracker:
        # Log validation metrics
        mlflow_tracker.log_metrics({
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_f1_weighted': val_metrics['f1_weighted'],
            'val_recall_macro': val_metrics['recall_macro'],
            'val_recall_weighted': val_metrics['recall_weighted']
        })

        # Log test metrics
        mlflow_tracker.log_metrics({
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'test_recall_macro': test_metrics['recall_macro'],
            'test_recall_weighted': test_metrics['recall_weighted']
        })

        # Log confusion matrix as artifact
        cm_dict = {
            'confusion_matrix': test_metrics['confusion_matrix'],
            'labels': ['low', 'medium', 'high']
        }
        mlflow_tracker.log_dict(cm_dict, 'confusion_matrix.json')

        # Log feature importance
        if hasattr(trainer.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': trainer.feature_cols,
                'importance': trainer.model.feature_importances_
            }).sort_values('importance', ascending=False)

            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                feature_importance.to_csv(f.name, index=False)
                mlflow_tracker.log_artifact(f.name, 'feature_importance.csv')

        # Log model to MLflow
        try:
            signature = create_model_signature(X_train, y_train)
            mlflow_tracker.log_model(
                model=trainer.model,
                artifact_path="model",
                registered_model_name="volatility_predictor",
                signature=signature,
                input_example=X_train[:5]
            )
            print("\n✓ Model logged to MLflow registry")
        except Exception as e:
            print(f"\n⚠ Failed to log model to MLflow: {e}")

    # Save model locally
    print("\n6. Saving model...")
    trainer.save_model(output_dir=args.output_dir)

    # End MLflow run
    if mlflow_tracker:
        mlflow_tracker.end_run(status="FINISHED")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")

    if mlflow_tracker:
        print("\n✓ View experiment in MLflow UI: http://localhost:5000")


if __name__ == '__main__':
    main()
