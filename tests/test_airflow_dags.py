"""Tests for Airflow DAG definitions."""
import pytest
from datetime import datetime
from airflow.models import DagBag


@pytest.mark.airflow
class TestDAGIntegrity:
    """Test Airflow DAG structure and configuration."""

    def test_dag_bag_import(self):
        """Test that all DAGs can be imported without errors."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)

        assert len(dag_bag.import_errors) == 0, \
            f"DAG import errors: {dag_bag.import_errors}"

    def test_required_dags_present(self):
        """Test that expected DAGs are defined."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)

        expected_dags = ['stock_grok_pipeline', 'ml_volatility_pipeline']

        for dag_id in expected_dags:
            assert dag_id in dag_bag.dags, f"Expected DAG '{dag_id}' not found"

    def test_dag_has_tags(self):
        """Test that DAGs have appropriate tags."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)

        for dag_id, dag in dag_bag.dags.items():
            assert dag.tags is not None, f"DAG '{dag_id}' has no tags"
            assert len(dag.tags) > 0, f"DAG '{dag_id}' has empty tags"

    def test_dag_has_owner(self):
        """Test that DAGs have an owner specified."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)

        for dag_id, dag in dag_bag.dags.items():
            assert dag.owner != 'airflow', \
                f"DAG '{dag_id}' should have custom owner, not default 'airflow'"

    def test_dag_has_retries(self):
        """Test that DAG tasks have retry configuration."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)

        for dag_id, dag in dag_bag.dags.items():
            for task in dag.tasks:
                assert task.retries is not None, \
                    f"Task '{task.task_id}' in DAG '{dag_id}' has no retry config"
                assert task.retries >= 0, \
                    f"Task '{task.task_id}' has negative retries"


@pytest.mark.airflow
class TestStockGrokPipeline:
    """Test stock_grok_pipeline DAG specifics."""

    def test_stock_pipeline_tasks(self):
        """Test that stock pipeline has expected tasks."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)
        dag = dag_bag.get_dag('stock_grok_pipeline')

        assert dag is not None, "stock_grok_pipeline DAG not found"

        expected_tasks = ['extract', 'enrich', 'transform', 'test']
        task_ids = [task.task_id for task in dag.tasks]

        for expected_task in expected_tasks:
            assert expected_task in task_ids, \
                f"Expected task '{expected_task}' not found in DAG"

    def test_stock_pipeline_schedule(self):
        """Test stock pipeline has appropriate schedule."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)
        dag = dag_bag.get_dag('stock_grok_pipeline')

        # Should have a schedule defined
        assert dag.schedule_interval is not None, \
            "stock_grok_pipeline should have a schedule"


@pytest.mark.airflow
class TestMLVolatilityPipeline:
    """Test ml_volatility_pipeline DAG specifics."""

    def test_ml_pipeline_tasks(self):
        """Test that ML pipeline has expected tasks."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)
        dag = dag_bag.get_dag('ml_volatility_pipeline')

        assert dag is not None, "ml_volatility_pipeline DAG not found"

        # Check for key ML tasks
        task_ids = [task.task_id for task in dag.tasks]

        # Should have training and prediction tasks
        assert any('train' in task_id.lower() for task_id in task_ids), \
            "ML pipeline should have a training task"
        assert any('predict' in task_id.lower() for task_id in task_ids), \
            "ML pipeline should have a prediction task"

    def test_ml_pipeline_dependencies(self):
        """Test ML pipeline has proper task dependencies."""
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)
        dag = dag_bag.get_dag('ml_volatility_pipeline')

        # Verify tasks have dependencies (not all parallel)
        has_dependencies = False
        for task in dag.tasks:
            if len(task.upstream_task_ids) > 0:
                has_dependencies = True
                break

        assert has_dependencies, \
            "ML pipeline should have task dependencies"
