#!/bin/bash
# Check if MLflow server is already running
if ! curl -s http://localhost:8080 > /dev/null; then
    echo "Starting MLflow server..."
    mlflow server \
        --host localhost \
        --port 8080 \
        --backend-store-uri sqlite:////content/drive/MyDrive/mlflow_experiments/mlflow.db \
        --default-artifact-root /content/drive/MyDrive/mlflow_experiments/artifacts \
        --dev &
    
    # Wait a moment for server to start
    sleep 5
    echo "MLflow server started at http://localhost:8080"
else
    echo "MLflow server already running"
fi