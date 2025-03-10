# Marxist Chat

A RAG (Retrieval Augmented Generation) chatbot built to interact with content from multiple communist theory sources. The application indexes articles from RSS feeds including communistusa.org and marxist.com, creating a searchable vector database that powers contextual responses to user queries.

## Core Functionality

- **Multi-Source Article Collection**: Downloads and processes articles from multiple RSS feeds with pagination support
- **Vector Database**: Creates and incrementally updates searchable embeddings of document content
- **Metadata Repository**: Efficiently tracks and manages document metadata for improved retrieval
- **Chat Interfaces**: Provides both command-line and web-based conversation options
- **Real-time Response Streaming**: Delivers token-by-token responses for better user experience
- **Source Attribution**: Cites specific articles used to generate each response

## Installation

### Prerequisites

- Python 3.8 or newer
- 2-10GB disk space for models (depending on model selection)
- 4GB+ RAM recommended

### Basic Installation

```bash
git clone https://github.com/yourusername/marxist-chat.git
cd marxist-chat
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your preferred settings
```

### Docker Installation

```bash
docker build -t marxist-chat .
docker run -p 8000:8000 -v ./posts_cache:/app/posts_cache -v ./vector_store:/app/vector_store -v ./logs:/app/logs marxist-chat
```

## Usage

### CLI Mode

```bash
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

### Web Server Mode

```bash
python src/app.py
```

Access the web interface at http://localhost:8000

## Configuration

The system can be configured through the `.env` file or via the CLI configuration menu.

### RSS Feed Configuration

The application now supports multiple feed sources with different pagination types:

```python
RSS_FEED_CONFIG = [
    {"url": "https://communistusa.org/feed", "pagination_type": "wordpress"},
    {"url": "https://marxist.com/index.php?format=feed", "pagination_type": "joomla", "limit_increment": 5},
    # Add more feeds with appropriate pagination_type and settings
]
```

### Models

#### Chat Models (smallest to largest):
- **Qwen 2.5 3B** (Default) - ~2GB download
- **Qwen 2.5 7B** - ~5GB download
- **Qwen 2.5 14B** - ~9GB download

#### Embedding Models:
- **BGE-M3** (Default)
- **GTE-Small**

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

## Key Features

### Incremental Vector Store Updates

The system can now update the vector store with new articles without rebuilding the entire database:

```bash
# CLI Option 10 or via API endpoint:
curl -X POST http://localhost:8000/api/v1/update-vector-store
```

### Enhanced Metadata Repository

Improved document tracking and retrieval with a dedicated metadata index:

- Fast document lookups by filename, title, or source
- Rich metadata extraction and preservation 
- Feed source tracking for multi-source setups

### Queue Management

Advanced queue management for the web API:
- Auto-placement of users in queue when server reaches capacity
- Real-time status updates and position notifications
- Configurable timeouts and concurrent user limits

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

Check the log files in the `logs/` directory for detailed error information.

## Data Storage

- `posts_cache/` - Organized by feed source, contains archived articles
- `vector_store/` - Contains vector database files
- `logs/` - Application logs
- `static/` - Frontend web interface files

## License

This project is licensed under the AGPLv3 License - see the LICENSE file for details.
