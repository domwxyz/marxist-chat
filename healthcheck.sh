#!/bin/bash
# Docker healthcheck script for API and LLM services

set -e

# Determine service type based on environment variable
if [ "${SERVICE_TYPE}" = "api" ]; then
  # API service healthcheck
  curl -f http://localhost:${PORT:-8000}/api/v1/healthcheck || exit 1
elif [ "${SERVICE_TYPE}" = "llm" ]; then
  # LLM service healthcheck
  curl -f http://localhost:${PORT:-5000}/health || exit 1
else
  # Default healthcheck
  if [ -f /app/src/api_adapter.py ]; then
    curl -f http://localhost:${PORT:-8000}/api/v1/healthcheck || exit 1
  else
    curl -f http://localhost:${PORT:-5000}/health || exit 1
  fi
fi

exit 0
