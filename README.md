# Marxist Chat

A RAG (Retrieval Augmented Generation) chatbot for articles from communistusa.org. This application allows users to chat with content from RSS feeds, with a focus on Revolutionary Communists of America (RCA) articles.

## Project Overview

The project has been restructured from a monolithic script into a modular application that supports:

1. A command-line interface (CLI) for local usage
2. A FastAPI web server for remote access
3. WebSocket support for real-time chat interactions with token-by-token streaming
4. A queueing system for handling concurrent users
5. A simple web interface for interacting with the chatbot
6. Kubernetes deployment for scaling and high availability

## Features

- **Real-time Streaming**: Token-by-token streaming via WebSockets for a responsive chat experience
- **Interruptible Queries**: Stop in-progress queries via WebSocket commands or REST API
- **Concurrent User Support**: Handles multiple users with a queueing system for overflow
- **Document Search**: Search and retrieve specific documents from the archive
- **Source Attribution**: Responses include relevant source information with titles, dates, and URLs
- **Metrics Collection**: System metrics for monitoring application health and performance
- **Scalable Architecture**: Designed for horizontal scaling with Kubernetes
- **Simple Web Interface**: Built-in frontend for easy interaction

## Requirements

- Python 3.8+
- Required Python packages:
  - llama-index-core >= 0.10.0
  - llama-index-llms-llama-cpp >= 0.1.0
  - llama-index-embeddings-huggingface >= 0.1.0
  - llama-index-vector-stores-chroma >= 0.1.0
  - chromadb >= 0.4.13
  - feedparser >= 6.0.0
  - chardet >= 5.0.0
  - fastapi >= 0.104.0
  - uvicorn >= 0.23.0
  - pydantic >= 2.0.0
  - python-dotenv >= 1.0.0
  - websockets >= 11.0.0
  - psutil >= 5.9.0 (optional, for metrics collection)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/marxist-chat.git
   cd marxist-chat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and edit it with your settings:
   ```bash
   cp .env.example .env
   ```

## Project Structure

```
marxist-chat/
├── src/                # Source code directory
│   ├── api/            # API endpoints and WebSocket handlers
│   │   ├── endpoints.py      # REST API endpoint implementations
│   │   ├── exceptions.py     # Exception handlers
│   │   ├── middleware.py     # Middleware for metrics
│   │   ├── metrics_endpoint.py # Metrics API endpoints
│   │   ├── router.py         # FastAPI router definitions
│   │   └── websocket.py      # WebSocket handling logic
│   ├── cli/            # Command-line interface components
│   │   ├── handlers.py       # Menu option handlers
│   │   └── menu.py           # CLI menu system
│   ├── core/           # Core application functionality
│   │   ├── feed_processor.py # RSS feed processing
│   │   ├── llm_manager.py    # LLM model management
│   │   ├── query_engine.py   # RAG query processing
│   │   └── vector_store.py   # Vector database management
│   ├── utils/          # Utility functions
│   │   ├── file_utils.py     # File management utilities
│   │   ├── logging_setup.py  # Logging configuration
│   │   ├── metadata_utils.py # Metadata extraction helpers
│   │   └── text_utils.py     # Text processing utilities
│   ├── app.py          # FastAPI application entry point
│   ├── config.py       # Application configuration
│   └── main.py         # CLI application entry point
├── kubernetes/         # Kubernetes deployment files
├── static/             # Static files for the web interface
│   └── index.html      # Frontend client HTML
├── posts_cache/        # Directory for cached RSS feed articles
├── vector_store/       # Directory for vector database storage
├── logs/               # Application logs
├── tests/              # Test directory
├── examples/           # Example clients and usage
├── Dockerfile          # Container definition
├── docker-compose.yml  # Docker Compose for development
└── requirements.txt    # Required Python packages
```

## Storage Directories

The application stores data in the following directories at the project root:

- `posts_cache/` - Cached RSS feed articles
- `vector_store/` - Vector database storage
- `logs/` - Application logs
- `static/` - Frontend web interface files

## Usage

### CLI Mode

Run the CLI version of the application:
```bash
python src/main.py
```

You will be presented with a menu interface with the following options:
1. Archive RSS Feed - Downloads articles from the RSS feed
2. Create Vector Store - Creates the vector database from archived articles
3. Load Vector Store - Loads the existing vector database
4. Load Chat - Starts the chat interface
5. Delete RSS Archive - Removes the downloaded articles
6. Delete Vector Store - Removes the vector database
7. Configuration - Adjust program settings
0. Exit - Quit the program

### Web Server Mode

Start the web server:
```bash
python src/app.py
```

This launches a FastAPI server on the configured host and port (default: http://0.0.0.0:8000).

Once the server is running, you can access the web interface by navigating to http://localhost:8000 in your browser.

### First-Time Setup with Web Interface

When using the application for the first time through the web interface:

1. Access the web interface at http://localhost:8000
2. Click "Connect" to establish a WebSocket connection
3. Click "Archive RSS Feed" to download articles
4. Click "Create Vector Store" to build the vector database (this may take several minutes)
5. Once the vector store is created, you can start asking questions in the chat interface

### Docker Deployment

Build and run using Docker:
```bash
docker build -t marxist-chat .
docker run -p 8000:8000 -v ./posts_cache:/app/posts_cache -v ./vector_store:/app/vector_store -v ./logs:/app/logs marxist-chat
```

Or use Docker Compose:
```bash
docker-compose up -d
```

### Kubernetes Deployment

For production deployment with Kubernetes, refer to the [Kubernetes Deployment Guide](kubernetes/README.md).

## API Endpoints

### Health and Status

- `GET /api/v1/healthcheck` - Check if the API is running
- `GET /api/v1/status` - Get system status information

### Queue Management

- `GET /api/v1/queue` - Get detailed queue status
- `POST /api/v1/queue/clear` - Admin endpoint to clear the waiting queue

### Archive and Vector Store Management

- `POST /api/v1/archive-rss` - Archive RSS feeds
- `POST /api/v1/create-vector-store` - Create vector store from archived documents
- `GET /api/v1/feed-stats` - Get feed statistics
- `GET /api/v1/vector-store-stats` - Get vector store statistics

### Document Access

- `GET /api/v1/documents/{document_id}` - Get a specific document by ID
- `GET /api/v1/documents/search` - Search for documents matching a query

### Model Information

- `GET /api/v1/model-info` - Get information about the current LLM model

### Query Management

- `POST /api/v1/query` - Process a query and get response with sources
- `POST /api/v1/query/{user_id}/stop` - Stop an in-progress query

### Service Management

- `POST /api/v1/service/restart` - Admin endpoint to restart service components

### Metrics

- `GET /api/v1/metrics` - Get detailed system metrics
- `GET /api/v1/metrics/summary` - Get a simplified metrics summary

### WebSocket Endpoints

- `WebSocket /api/v1/ws/chat/{user_id}` - WebSocket endpoint for chat interface
- `WebSocket /api/v1/ws/chat` - WebSocket endpoint with auto-generated ID

For detailed API documentation, see [API.md](src/api/API.md).

## Configuration

Edit `.env` or modify `src/config.py` to customize:

- RSS feed URLs
- Embedding and LLM models
- Server settings (host, port, concurrency limits)
- System prompt for the LLM
- Storage directories for cache and vector store

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HOST | Server hostname | 0.0.0.0 |
| PORT | Server port | 8000 |
| DEBUG | Enable debug mode | False |
| MAX_CONCURRENT_USERS | Maximum concurrent users | 30 |
| QUEUE_TIMEOUT | Queue timeout in seconds | 300 |
| REQUEST_TIMEOUT | Request timeout in seconds | 120 |
| CURRENT_LLM | URL to LLM model | Qwen2.5-3B-Instruct GGUF |
| CURRENT_EMBED | Embedding model name | BAAI/bge-m3 |
| NUM_THREADS | Number of threads for LLM inference | 4 |
| TEMPERATURE | Temperature for LLM responses | 0.2 |
| LOG_LEVEL | Logging level | INFO |

## Models

The application supports multiple models:

### Chat Models (smallest to largest):
- Qwen 2.5 3B (Default) - ~2GB download
- Qwen 2.5 7B - ~5GB download
- Qwen 2.5 14B - ~9GB download

### Embedding Models:
- BGE-M3 (Default)
- GTE-Small

## Web Interface

The application includes a simple web interface for interacting with the chatbot. The interface provides:

- Real-time WebSocket connection to the chatbot
- Token-by-token streaming for responsive chat experience
- Ability to interrupt long-running queries
- Buttons for archiving RSS feeds and creating the vector store
- Status checking capabilities
- Source attribution for responses

## Scalability

The application supports three scaling methods:

1. **Single Server**: Handles 20-30 concurrent users with queuing
2. **Docker Compose**: Useful for development and single-machine deployment
3. **Kubernetes**: For production deployment with horizontal scaling

## WebSocket Message Format

### Client to Server Messages

**Query Message:**
```json
{
  "message": "What is the communist position on healthcare?"
}
```

**Stop Query Command:**
```json
{
  "command": "stop_query"
}
```

### Server to Client Messages

**System Messages:**
```json
{
  "type": "system",
  "message": "Connected to chat service"
}
```

**Queue Messages:**
```json
{
  "type": "queue",
  "position": 3,
  "message": "You are #3 in queue. Estimated wait time: ~6 minutes"
}
```

**Status Messages:**
```json
{
  "type": "status",
  "message": "Processing your query..."
}
```

**Stream Messages:**
```json
{
  "type": "stream_token",
  "data": "token"
}
```

**Query Stopped Messages:**
```json
{
  "type": "query_stopped",
  "message": "Query was stopped by user request."
}
```

**Source Messages:**
```json
{
  "type": "sources",
  "data": [
    {
      "title": "Healthcare as a Human Right",
      "date": "2023-02-15",
      "url": "https://communistusa.org/2023/02/healthcare-human-right",
      "excerpt": "Access to healthcare is a fundamental human right that should not be commodified..."
    }
  ]
}
```

## Customization

### System Prompt

You can customize the system prompt by modifying the `SYSTEM_PROMPT` variable in your `.env` file or `src/config.py`. The system prompt sets the behavior and personality of the chatbot.

### Web Interface

The web interface can be customized by modifying the `static/index.html` file. This includes styles, layout, and functionality.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

- **WebSocket Connection Failed**: Ensure your server is running and check for firewall issues.
- **No Responses to Questions**: Verify that both RSS archive and vector store have been created.
- **Model Loading Errors**: Check disk space and ensure the model URL is accessible.
- **Slow Response Times**: Check the NUM_THREADS setting and consider using a smaller model.
- **Interrupted Queries Not Stopping**: Ensure WebSocket connection is stable and backend is responsive.

### Logs

Check the log files in the `logs/` directory for detailed error information.

## Performance Considerations

- The application is resource-intensive, particularly during vector store creation and query processing.
- For production deployments, consider server specifications with at least:
  - 4+ CPU cores
  - 8GB+ RAM
  - SSD storage for faster vector operations
  - Adequate bandwidth for model downloads

## License

This project is licensed under the AGPLv3 License - see the LICENSE file for details.