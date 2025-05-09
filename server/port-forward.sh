#!/bin/bash

# Define variables
SERVICE_NAME="image-platform-service"
NAMESPACE="default"  # Change if your service is in a different namespace
LOCAL_PORT=30080
REMOTE_PORT=80
LOG_FILE="portforward.log"

# Check if port-forward is already running
if pgrep -f "kubectl port-forward svc/$SERVICE_NAME $LOCAL_PORT:$REMOTE_PORT" > /dev/null; then
    echo "Port-forward already running for $SERVICE_NAME on port $LOCAL_PORT"
else
    echo "Starting port-forward..."
    nohup kubectl port-forward svc/$SERVICE_NAME $LOCAL_PORT:$REMOTE_PORT -n $NAMESPACE > "$LOG_FILE" 2>&1 &
    echo "Port-forward started. Output is being logged to $LOG_FILE"
fi
