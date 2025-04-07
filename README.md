# Marxist Chat

A distributed RAG (Retrieval Augmented Generation) chatbot built to interact with content from multiple communist theory sources. The application indexes articles from RSS feeds including communistusa.org and marxist.com, creating a searchable vector database that powers contextual responses to user queries.

## Core Functionality

- **Multi-Source Article Collection**: Downloads and processes articles from multiple RSS feeds with pagination support
- **Vector Database**: Creates and incrementally updates searchable embeddings of document content
- **Metadata Repository**: Efficiently tracks and manages document metadata for improved retrieval
- **Distributed Architecture**: Supports horizontal scaling with multiple LLM instances
- **Load Balancing**: Efficiently distributes requests across multiple service instances
- **Chat Interfaces**: Provides both command-line and web-based conversation options
- **Real-time Response Streaming**: Delivers token-by-token responses for better user experience
- **Source Attribution**: Cites specific articles used to generate each response

## Architecture

Marxist Chat is built as a distributed system with the following components:

```
                   ┌─────────────────┐
                   │                 │
Users ────────────▶│  Nginx Proxy    │
                   │  (Load Balancer)│
                   │                 │
                   └────────┬────────┘
                            │
                            ▼
         ┌──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────┐               ┌─────────────────┐
│  API Container 1│               │  API Container 2│
│ (Queue Manager  │               │ (Queue Manager  │
│  & WebSockets)  │               │  & WebSockets)  │
└────────┬────────┘               └────────┬────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐               ┌─────────────────┐
│  LLM Container 1│               │  LLM Container 2│
│  (Qwen LLM)     │               │  (Qwen LLM)     │
└─────────────────┘               └─────────────────┘
```

- **API Service**: Handles WebSockets, user connections, and queuing
- **LLM Service**: Dedicated to running LlamaCPP inference
- **Redis**: Centralized state management and cross-instance communication
- **Nginx**: Load balancing and request routing

## Installation

### Prerequisites

- Docker and Docker Compose
- 4GB+ RAM (8GB recommended for multiple instances)
- 2-4 vCPU recommended
- 5GB+ disk space for models and vector store

### Docker Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/marxist-chat.git
cd marxist-chat

# Initialize the environment
chmod +x initialize.sh
./initialize.sh

# Start the services
docker-compose up -d
```

### Scaling the Deployment

To scale the number of API or LLM instances:

```bash
# Scale to 3 API instances and 2 LLM instances
docker-compose up -d --scale api=3 --scale llm=2
```

## Configuration

The system can be configured through the `.env` file or via environment variables.

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| HOST | Server hostname | 0.0.0.0 |
| PORT | Server port | 8000 |
| MAX_CONCURRENT_USERS | Maximum concurrent users | 30 |
| CURRENT_LLM | URL to LLM model | Qwen2.5-1.5B-Instruct GGUF |
| CURRENT_EMBED | Embedding model name | BAAI/bge-m3 |
| NUM_THREADS | Number of threads for LLM inference | 2 |
| DISTRIBUTED_MODE | Enable distributed architecture | true |
| REDIS_HOST | Redis server hostname | redis |
| REDIS_PORT | Redis server port | 6379 |

### RSS Feed Configuration

The application supports multiple feed sources with different pagination types:

```python
RSS_FEED_CONFIG = [
    {"url": "https://communistusa.org/feed", "pagination_type": "wordpress"},
    {"url": "https://marxist.com/index.php?format=feed", "pagination_type": "joomla", "limit_increment": 5},
    {"url": "https://communist.red/feed", "pagination_type": "wordpress"},
    # Add more feeds with appropriate pagination_type and settings
]
```

### Models

#### Chat Models (smallest to largest):
- **Qwen 2.5 1.5B** (Default) - ~2GB download
- **Qwen 2.5 3B** - ~3GB download
- **Qwen 2.5 7B** - ~5GB download
- **Qwen 2.5 14B** - ~9GB download

#### Embedding Models:
- **BGE-M3** (Default)
- **GTE-Small**

## Usage

### Setup via CLI

Once the containers are running, you can set up the RSS archive and vector store using the CLI:

```bash
# Connect to an API container
docker-compose exec api1 bash

# Run the CLI for first-time setup
python src/main.py
```

Menu options:
1. **Archive RSS Feed** - Downloads articles from configured RSS feeds
2. **Create Vector Store** - Creates vector database from archived articles
3. **Load Vector Store** - Loads existing vector database
4. **Load Chat** - Starts chat interface
5. **Delete RSS Archive** - Removes downloaded articles
6. **Delete Vector Store** - Removes vector database
7. **Configuration** - Adjust program settings
8. **Test LLM** - Verify LLM functionality
9. **Rebuild Metadata Index** - Refresh metadata repository
10. **Update Vector Store** - Add new articles without rebuilding

### Web Interface

Access the web interface at http://localhost:8000 (or your configured domain)

## Directory Structure

```
marxist-chat/
├── docker-compose.yml     # Docker Compose configuration
├── nginx.conf             # Nginx configuration
├── Dockerfile.api         # API service Dockerfile
├── Dockerfile.llm         # LLM service Dockerfile
├── .env                   # Environment configuration
├── initialize.sh          # Initialization script
├── start.sh               # Service startup script
├── healthcheck.sh         # Health check script
├── api_adapter.py         # API service adapter
├── llm_service.py         # LLM service implementation
├── src/                   # Application source code
│   ├── api/               # API implementation
│   ├── cli/               # Command-line interface
│   ├── core/              # Core functionality
│   │   ├── feed_processor.py
│   │   ├── llm_manager.py
│   │   ├── metadata_repository.py
│   │   ├── query_engine.py
│   │   └── vector_store.py
│   ├── utils/             # Utility functions
│   │   ├── redis_helper.py
│   │   └── ...
│   ├── app.py             # Web server
│   └── main.py            # CLI entry point
├── models/                # LLM model storage
├── posts_cache/           # Downloaded articles
├── vector_store/          # Vector database
├── logs/                  # Application logs
└── static/                # Web interface files
```

## Key Features

### Distributed Architecture

- **Horizontal Scaling**: Add more API and LLM containers as needed
- **Resource Isolation**: Each LLM container has dedicated CPU/memory resources 
- **Fault Tolerance**: If one container fails, others continue serving requests
- **Load Distribution**: Efficiently routes traffic based on container availability

### Queue Management

Advanced queue management across instances:
- Auto-placement of users in queue when server reaches capacity
- Real-time status updates and position notifications
- Configurable timeouts and concurrent user limits

### Incremental Vector Store Updates

The system can update the vector store with new articles without rebuilding the entire database:

```bash
# CLI Option 10 or via API endpoint:
curl -X POST http://localhost:8000/api/v1/update-vector-store
```

## API Documentation

Comprehensive API endpoints are available for programmatic integration. See [API Documentation](src/api/API.md) for details on:

- WebSocket connections for streaming chat
- REST endpoints for system management
- Document search and retrieval
- Queue monitoring and control

## Troubleshooting

### Common Issues

- **Missing Articles**: Ensure all RSS feeds are correctly configured with appropriate pagination types
- **Slow Responses**: Adjust NUM_THREADS setting or use a smaller model
- **Search Quality Issues**: Rebuild the metadata index with option 9
- **New Content Not Appearing**: Use option 10 to update the vector store with latest articles
- **Connection Issues**: Check logs with `docker-compose logs -f` for details

## License

This project is licensed under the AGPLv3 License - see the LICENSE file for details.
