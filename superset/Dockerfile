FROM apache/superset:latest

USER root

# Install system dependencies and PostgreSQL client
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

USER superset

# Install PostgreSQL driver
RUN pip install --no-cache-dir psycopg2-binary
