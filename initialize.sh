#!/bin/bash
# First-time initialization script for Marxist Chat

set -e

echo "Initializing Marxist Chat distributed environment..."

# Create necessary directories
mkdir -p models posts_cache vector_store logs llm_logs static

# Copy static files if not already present
if [ ! -f static/index.html ]; then
  echo "Copying static files..."
  cp -r src/static/* static/ 2>/dev/null || true
fi

# Create environment file if not present
if [ ! -f .env ]; then
  echo "Creating .env file from template..."
  cat > .env << EOL
# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Number of concurrent connections
MAX_CONCURRENT_USERS=3
QUEUE_TIMEOUT=300
REQUEST_TIMEOUT=120

# LLM configuration
CURRENT_LLM=https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
CURRENT_EMBED=BAAI/bge-m3
NUM_THREADS=1
TEMPERATURE=0.2

# Redis configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Distributed mode
DISTRIBUTED_MODE=true

# Logging
LOG_LEVEL=INFO
EOL
fi

# Check if Docker Compose file exists
if [ ! -f docker-compose.yml ]; then
  echo "Docker Compose file not found. Please copy the docker-compose.yml file to this directory."
  exit 1
fi

# Make scripts executable
chmod +x *.sh 2>/dev/null || true

echo "Initialization complete!"
echo ""
echo "Next steps:"
echo "1. Review and edit .env file if needed"
echo "2. Start the services with: docker-compose up -d"
echo "3. Check logs with: docker-compose logs -f"
echo ""
echo "To archive RSS feeds and create vector store:"
echo "docker-compose exec api1 python src/main.py"
echo ""
echo "For more information, see the README.md file."
