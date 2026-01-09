#!/usr/bin/env python3
"""CLI for managing the Stock-Grok ETL pipeline."""

import argparse
import os
import subprocess
import sys

AVAILABLE_MODELS = [
    ("grok-3", "Latest full Grok model (default)"),
    ("grok-3-mini", "Smaller, faster variant"),
    ("grok-3-fast", "Optimized for speed"),
]

DEFAULT_MODEL = "grok-3"


def get_current_model():
    """Get the currently configured Grok model."""
    return os.getenv("GROK_MODEL", DEFAULT_MODEL)


def run_command(cmd, env=None):
    """Run a shell command and return the result."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(cmd, shell=True, env=full_env)
    return result.returncode


def cmd_models(args):
    """List available Grok models."""
    current = get_current_model()
    print("Available Grok models:\n")
    for model, desc in AVAILABLE_MODELS:
        marker = " *" if model == current else ""
        print(f"  {model:<15} {desc}{marker}")
    print(f"\n* = currently selected (via GROK_MODEL env var)")
    print(f"\nTo change: export GROK_MODEL=<model-name>")


def cmd_run(args):
    """Run the pipeline with optional model override."""
    model = args.model or get_current_model()
    print(f"Running pipeline with Grok model: {model}")

    env = {"GROK_MODEL": model}
    return run_command("docker-compose up -d", env=env)


def cmd_status(args):
    """Check pipeline status."""
    print("Checking service status...\n")
    run_command("docker-compose ps")

    print("\nCurrent configuration:")
    print(f"  GROK_MODEL: {get_current_model()}")

    # Check if XAI_API_KEY is set
    if os.getenv("XAI_API_KEY"):
        print("  XAI_API_KEY: [set]")
    else:
        print("  XAI_API_KEY: [not set] - add to .env file")


def cmd_trigger(args):
    """Trigger the Airflow DAG."""
    model = args.model or get_current_model()
    print(f"Triggering stock_grok_pipeline with model: {model}")

    # Set the model env var in the container
    cmd = (
        f'docker-compose exec -e GROK_MODEL={model} airflow-webserver '
        f'airflow dags trigger stock_grok_pipeline'
    )
    return run_command(cmd)


def cmd_logs(args):
    """View Airflow scheduler logs."""
    service = args.service or "airflow-scheduler"
    cmd = f"docker-compose logs -f {service}"
    if args.tail:
        cmd = f"docker-compose logs --tail={args.tail} {service}"
    return run_command(cmd)


def cmd_stop(args):
    """Stop all services."""
    print("Stopping services...")
    return run_command("docker-compose down")


def cmd_restart(args):
    """Restart services with optional model change."""
    model = args.model or get_current_model()
    print(f"Restarting services with Grok model: {model}")

    env = {"GROK_MODEL": model}
    run_command("docker-compose down", env=env)
    return run_command("docker-compose up -d", env=env)


def main():
    parser = argparse.ArgumentParser(
        description="CLI for managing the Stock-Grok ETL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # models command
    models_parser = subparsers.add_parser("models", help="List available Grok models")
    models_parser.set_defaults(func=cmd_models)

    # run command
    run_parser = subparsers.add_parser("run", help="Start the pipeline services")
    run_parser.add_argument(
        "--model", "-m",
        choices=[m[0] for m in AVAILABLE_MODELS],
        help="Grok model to use"
    )
    run_parser.set_defaults(func=cmd_run)

    # status command
    status_parser = subparsers.add_parser("status", help="Check pipeline status")
    status_parser.set_defaults(func=cmd_status)

    # trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Trigger the Airflow DAG")
    trigger_parser.add_argument(
        "--model", "-m",
        choices=[m[0] for m in AVAILABLE_MODELS],
        help="Grok model to use for this run"
    )
    trigger_parser.set_defaults(func=cmd_trigger)

    # logs command
    logs_parser = subparsers.add_parser("logs", help="View service logs")
    logs_parser.add_argument(
        "--service", "-s",
        choices=["airflow-scheduler", "airflow-webserver", "postgres", "metabase"],
        help="Service to view logs for (default: airflow-scheduler)"
    )
    logs_parser.add_argument(
        "--tail", "-n",
        type=int,
        help="Number of lines to show"
    )
    logs_parser.set_defaults(func=cmd_logs)

    # stop command
    stop_parser = subparsers.add_parser("stop", help="Stop all services")
    stop_parser.set_defaults(func=cmd_stop)

    # restart command
    restart_parser = subparsers.add_parser("restart", help="Restart services")
    restart_parser.add_argument(
        "--model", "-m",
        choices=[m[0] for m in AVAILABLE_MODELS],
        help="Grok model to use"
    )
    restart_parser.set_defaults(func=cmd_restart)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    result = args.func(args)
    sys.exit(result or 0)


if __name__ == "__main__":
    main()
