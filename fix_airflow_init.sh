#!/bin/bash

# Fix for ContainerConfig error with airflow-init
# This script removes the problematic container and rebuilds it

echo "Stopping and removing airflow-init container..."
docker-compose rm -f airflow-init

echo "Removing the problematic container if it exists..."
docker rm -f 889f8823d4b4_etl-crypto_airflow-init_1 2>/dev/null || true

echo "Removing any containers with airflow-init in the name..."
docker ps -a | grep airflow-init | awk '{print $1}' | xargs -r docker rm -f

echo "Rebuilding the airflow image..."
docker-compose build airflow-init

echo "Now try running: docker-compose up airflow-init"

