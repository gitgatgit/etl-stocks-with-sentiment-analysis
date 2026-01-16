"""Airflow utilities for monitoring and alerting."""

from airflow.utils.alerts import (
    slack_failure_callback,
    slack_success_callback,
    slack_sla_miss_callback,
    pagerduty_failure_callback,
    pagerduty_success_callback,
    teams_failure_callback,
    teams_success_callback,
)
