"""
Airflow DAG alerting utilities for third-party integrations.

Supports:
- Slack (webhook-based)
- PagerDuty (events API)
- Microsoft Teams (webhook-based)
"""
import json
import logging
import os
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


def _send_webhook(url: str, payload: dict, headers: Optional[dict] = None) -> bool:
    """Send a JSON payload to a webhook URL."""
    if not url:
        logger.warning("Webhook URL not configured, skipping notification")
        return False

    default_headers = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)

    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers=default_headers, method="POST")
        with urlopen(req, timeout=30) as response:
            logger.info(f"Webhook sent successfully: {response.status}")
            return True
    except HTTPError as e:
        logger.error(f"Webhook HTTP error: {e.code} - {e.reason}")
    except URLError as e:
        logger.error(f"Webhook URL error: {e.reason}")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
    return False


def _format_task_context(context: dict) -> dict:
    """Extract relevant task info from Airflow context."""
    task_instance = context.get("task_instance")
    dag_run = context.get("dag_run")

    return {
        "dag_id": context.get("dag").dag_id if context.get("dag") else "unknown",
        "task_id": task_instance.task_id if task_instance else "unknown",
        "execution_date": str(context.get("execution_date", "unknown")),
        "run_id": dag_run.run_id if dag_run else "unknown",
        "try_number": task_instance.try_number if task_instance else 0,
        "log_url": task_instance.log_url if task_instance else None,
        "exception": str(context.get("exception", "")) if context.get("exception") else None,
    }


# =============================================================================
# Slack Integration
# =============================================================================

def slack_failure_callback(context: dict) -> None:
    """Send Slack notification on task failure."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set, skipping Slack notification")
        return

    info = _format_task_context(context)

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Airflow Task Failed",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*DAG:*\n{info['dag_id']}"},
                {"type": "mrkdwn", "text": f"*Task:*\n{info['task_id']}"},
                {"type": "mrkdwn", "text": f"*Execution Date:*\n{info['execution_date']}"},
                {"type": "mrkdwn", "text": f"*Try Number:*\n{info['try_number']}"},
            ]
        },
    ]

    if info["exception"]:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Error:*\n```{info['exception'][:500]}```"
            }
        })

    if info["log_url"]:
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Logs"},
                    "url": info["log_url"],
                    "style": "danger"
                }
            ]
        })

    payload = {
        "text": f"Task {info['dag_id']}.{info['task_id']} failed",
        "blocks": blocks
    }

    _send_webhook(webhook_url, payload)


def slack_success_callback(context: dict) -> None:
    """Send Slack notification on DAG success (use on last task only)."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return

    info = _format_task_context(context)

    payload = {
        "text": f"DAG {info['dag_id']} completed successfully",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*DAG `{info['dag_id']}` completed successfully*\nExecution: {info['execution_date']}"
                }
            }
        ]
    }

    _send_webhook(webhook_url, payload)


def slack_sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis) -> None:
    """Send Slack notification on SLA miss."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return

    task_names = [t.task_id for t in task_list] if task_list else []

    payload = {
        "text": f"SLA Miss in DAG {dag.dag_id}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "SLA Miss Alert", "emoji": True}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n{dag.dag_id}"},
                    {"type": "mrkdwn", "text": f"*Tasks:*\n{', '.join(task_names)}"},
                ]
            }
        ]
    }

    _send_webhook(webhook_url, payload)


# =============================================================================
# PagerDuty Integration
# =============================================================================

def pagerduty_failure_callback(context: dict) -> None:
    """Send PagerDuty alert on task failure."""
    routing_key = os.getenv("PAGERDUTY_ROUTING_KEY")
    if not routing_key:
        logger.warning("PAGERDUTY_ROUTING_KEY not set, skipping PagerDuty notification")
        return

    info = _format_task_context(context)

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "dedup_key": f"{info['dag_id']}-{info['task_id']}-{info['run_id']}",
        "payload": {
            "summary": f"Airflow task failed: {info['dag_id']}.{info['task_id']}",
            "severity": "error",
            "source": "airflow",
            "custom_details": {
                "dag_id": info["dag_id"],
                "task_id": info["task_id"],
                "execution_date": info["execution_date"],
                "try_number": info["try_number"],
                "error": info["exception"],
            }
        },
        "links": [{"href": info["log_url"], "text": "Airflow Logs"}] if info["log_url"] else []
    }

    _send_webhook("https://events.pagerduty.com/v2/enqueue", payload)


def pagerduty_success_callback(context: dict) -> None:
    """Resolve PagerDuty incident on task success."""
    routing_key = os.getenv("PAGERDUTY_ROUTING_KEY")
    if not routing_key:
        return

    info = _format_task_context(context)

    payload = {
        "routing_key": routing_key,
        "event_action": "resolve",
        "dedup_key": f"{info['dag_id']}-{info['task_id']}-{info['run_id']}",
    }

    _send_webhook("https://events.pagerduty.com/v2/enqueue", payload)


# =============================================================================
# Microsoft Teams Integration
# =============================================================================

def teams_failure_callback(context: dict) -> None:
    """Send Microsoft Teams notification on task failure."""
    webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("TEAMS_WEBHOOK_URL not set, skipping Teams notification")
        return

    info = _format_task_context(context)

    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "FF0000",
        "summary": f"Airflow Task Failed: {info['dag_id']}.{info['task_id']}",
        "sections": [
            {
                "activityTitle": "Airflow Task Failed",
                "facts": [
                    {"name": "DAG", "value": info["dag_id"]},
                    {"name": "Task", "value": info["task_id"]},
                    {"name": "Execution Date", "value": info["execution_date"]},
                    {"name": "Try Number", "value": str(info["try_number"])},
                ],
                "markdown": True
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": "View Logs",
                "targets": [{"os": "default", "uri": info["log_url"]}]
            }
        ] if info["log_url"] else []
    }

    if info["exception"]:
        payload["sections"][0]["text"] = f"**Error:** {info['exception'][:300]}"

    _send_webhook(webhook_url, payload)


def teams_success_callback(context: dict) -> None:
    """Send Microsoft Teams notification on DAG success."""
    webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
    if not webhook_url:
        return

    info = _format_task_context(context)

    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "00FF00",
        "summary": f"DAG {info['dag_id']} completed successfully",
        "sections": [
            {
                "activityTitle": "DAG Completed Successfully",
                "facts": [
                    {"name": "DAG", "value": info["dag_id"]},
                    {"name": "Execution Date", "value": info["execution_date"]},
                ],
                "markdown": True
            }
        ]
    }

    _send_webhook(webhook_url, payload)
