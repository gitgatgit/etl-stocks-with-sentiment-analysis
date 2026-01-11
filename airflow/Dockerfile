FROM apache/airflow:2.8.0-python3.11

USER root
RUN apt-get update && apt-get install -y git && \
    mkdir -p /opt/airflow/logs && \
    chown -R airflow:root /opt/airflow/logs && \
    chmod -R 775 /opt/airflow/logs

USER airflow

# Copy ML requirements first
COPY --chown=airflow:root ml/requirements.txt /tmp/ml-requirements.txt

# Install all dependencies
RUN pip install --no-cache-dir \
    "protobuf<5.0.0dev,>=3.19.5" \
    yfinance \
    openai \
    dbt-core \
    dbt-postgres && \
    pip install --no-cache-dir -r /tmp/ml-requirements.txt && \
    rm /tmp/ml-requirements.txt
