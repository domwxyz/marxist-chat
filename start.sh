#!/bin/bash
# Startup script for services

set -e

# Determine which service to start based on environment variable
if [ "${SERVICE_TYPE}" = "api" ]; then
  echo "Starting API service..."
  # Ensure Redis is available
  max_retries=30
  counter=0
  until redis-cli -h ${REDIS_HOST:-redis} ping > /dev/null 2>&1; do
    sleep 1
    counter=$((counter + 1))
    if [ $counter -ge $max_retries ]; then
      echo "Redis not available after $max_retries attempts. Exiting."
      exit 1
    fi
    echo "Waiting for Redis... ($counter/$max_retries)"
  done
  
  # Set distributed mode
  export DISTRIBUTED_MODE=true
  
  # Start API service
  python src/api_adapter.py
  
elif [ "${SERVICE_TYPE}" = "llm" ]; then
  echo "Starting LLM service..."
  
  # Check if model exists, download if needed
  MODEL_FILENAME=$(basename ${CURRENT_LLM})
  MODEL_PATH="/app/models/${MODEL_FILENAME}"
  
  if [ ! -f "${MODEL_PATH}" ]; then
    echo "Downloading model from ${CURRENT_LLM}..."
    mkdir -p /app/models
    curl -L -o "${MODEL_PATH}" "${CURRENT_LLM}"
  fi
  
  # Set environment variables
  export LOCAL_MODEL_PATH="${MODEL_PATH}"
  
  # Start LLM service
  python src/llm_service.py
  
else
  echo "No service type specified. Starting in auto-detect mode..."
  
  # Check if running in a Docker environment
  if [ -f /.dockerenv ]; then
    # Auto-detect based on installed files
    if [ -f /app/src/api_adapter.py ]; then
      echo "Detected API service."
      export SERVICE_TYPE=api
      exec $0
    elif [ -f /app/src/llm_service.py ]; then
      echo "Detected LLM service."
      export SERVICE_TYPE=llm
      exec $0
    else
      echo "Could not determine service type. Starting in standalone mode."
      python src/app.py
    fi
  else
    # Not in Docker, start in standalone mode
    echo "Starting in standalone mode..."
    python src/app.py
  fi
fi
