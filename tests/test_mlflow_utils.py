"""Tests for MLflow utilities module."""
import pytest
from unittest.mock import MagicMock, patch
from ml.mlflow_utils import MLflowTracker
import pandas as pd


@pytest.mark.unit
class TestMLflowTracker:
    """Test MLflow tracker functionality."""

    @patch('ml.mlflow_utils.mlflow')
    @patch('ml.mlflow_utils.MlflowClient')
    def test_log_params(self, mock_client, mock_mlflow):
        """Test logging parameters."""
        tracker = MLflowTracker()
        params = {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}

        tracker.log_params(params)

        assert mock_mlflow.log_param.call_count == 3

    @patch('ml.mlflow_utils.mlflow')
    @patch('ml.mlflow_utils.MlflowClient')
    def test_log_metrics(self, mock_client, mock_mlflow):
        """Test logging all metrics including accuracy, f1, and recall."""
        tracker = MLflowTracker()
        metrics = {
            'accuracy': 0.85,
            'f1_macro': 0.82,
            'f1_weighted': 0.83,
            'recall_macro': 0.80,
            'recall_weighted': 0.81
        }

        tracker.log_metrics(metrics)

        # Verify all 5 metrics are logged
        assert mock_mlflow.log_metric.call_count == 5

        # Check that all metrics are included
        calls = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        assert 'accuracy' in calls
        assert 'f1_macro' in calls
        assert 'f1_weighted' in calls
        assert 'recall_macro' in calls
        assert 'recall_weighted' in calls

    @patch('ml.mlflow_utils.mlflow')
    @patch('ml.mlflow_utils.MlflowClient')
    def test_log_validation_and_test_metrics(self, mock_client, mock_mlflow):
        """Test logging separate validation and test metrics with all metrics."""
        tracker = MLflowTracker()

        val_metrics = {
            'val_accuracy': 0.84,
            'val_f1_macro': 0.81,
            'val_f1_weighted': 0.82,
            'val_recall_macro': 0.79,
            'val_recall_weighted': 0.80
        }

        test_metrics = {
            'test_accuracy': 0.85,
            'test_f1_macro': 0.82,
            'test_f1_weighted': 0.83,
            'test_recall_macro': 0.80,
            'test_recall_weighted': 0.81
        }

        tracker.log_metrics(val_metrics)
        tracker.log_metrics(test_metrics)

        # Total of 10 metrics logged (5 val + 5 test)
        assert mock_mlflow.log_metric.call_count == 10

        # Check all validation metrics
        calls = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        assert 'val_accuracy' in calls
        assert 'val_f1_macro' in calls
        assert 'val_f1_weighted' in calls
        assert 'val_recall_macro' in calls
        assert 'val_recall_weighted' in calls

        # Check all test metrics
        assert 'test_accuracy' in calls
        assert 'test_f1_macro' in calls
        assert 'test_f1_weighted' in calls
        assert 'test_recall_macro' in calls
        assert 'test_recall_weighted' in calls

    @patch('ml.mlflow_utils.mlflow')
    @patch('ml.mlflow_utils.MlflowClient')
    def test_start_run_with_tags(self, mock_client, mock_mlflow):
        """Test starting run with tags."""
        tracker = MLflowTracker()
        tags = {'model_type': 'xgboost'}

        tracker.start_run(run_name='test_run', tags=tags)

        mock_mlflow.start_run.assert_called_once_with(run_name='test_run', tags=tags)

    @patch('ml.mlflow_utils.mlflow')
    @patch('ml.mlflow_utils.MlflowClient')
    def test_end_run(self, mock_client, mock_mlflow):
        """Test ending MLflow run."""
        tracker = MLflowTracker()

        tracker.end_run(status='FINISHED')

        mock_mlflow.end_run.assert_called_once_with(status='FINISHED')
