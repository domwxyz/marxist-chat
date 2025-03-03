# Marxist Chat

## What is Marxist Chat?

Marxist Chat is a RAG (Retrieval Augmented Generation) chatbot that allows users to interact with content primarily from the RSS feed at communistusa.org. The application indexes articles, creates a searchable vector database, and uses this knowledge to generate responses to user queries.

The system retrieves relevant document sections from the vector store, feeds them as context to a language model, and generates responses that include direct attribution to source materials. This ensures all responses are grounded in the actual content from the indexed articles.

## Core Functionality

- **Article Collection**: Downloads and processes articles from RSS feeds
- **Vector Database**: Creates searchable embeddings of document content
- **Chat Interface**: Provides both command-line and web-based interfaces
- **Real-time Responses**: Streams token-by-token responses for better user experience
- **Source Attribution**: Cites specific articles used to generate each response

Users can query about specific communist topics, and the system will provide information based solely on the indexed content, with references to the original articles.

# Installation Instructions

## Prerequisites

- Python 3.8 or newer
- 2-10GB disk space for models (depending on model selection)
- 4GB+ RAM recommended

## Basic Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/marxist-chat.git
   cd marxist-chat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy and configure the environment file:
   ```bash
   cp .env.example .env
   # Edit .env with appropriate settings
   ```

## Docker Installation

If you prefer using Docker:

```bash
docker build -t marxist-chat .
docker run -p 8000:8000 -v ./posts_cache:/app/posts_cache -v ./vector_store:/app/vector_store -v ./logs:/app/logs marxist-chat
```

## First-Time Setup

After installation, you'll need to:

1. Run the application:
   ```bash
   # CLI mode
   python src/main.py
   
   # Web server mode
   python src/app.py
   ```

2. For CLI mode, follow the menu prompts to:
   - Archive RSS feeds (Option 1)
   - Create the vector store (Option 2)
   - Load the vector store (Option 3)
   - Start the chat interface (Option 4)

3. For web mode, access http://localhost:8000 and use the web interface to perform the same setup steps.

The first-time setup will download the selected language model and embedding model, which may take several minutes depending on your internet connection.

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

## Requirements

- Python 3.8+
- Required Python packages:
    llama-index-core>=0.10.0
    llama-index-llms-llama-cpp>=0.1.0
    llama-index-embeddings-huggingface>=0.1.0
    llama-index-vector-stores-chroma>=0.1.0
    chromadb>=0.4.13
    feedparser>=6.0.0
    chardet>=5.0.0
    fastapi>=0.104.0
    uvicorn>=0.23.0
    pydantic>=2.0.0
    python-dotenv>=1.0.0
    websockets>=11.0.0
    psutil>=5.9.0  # Optional, for metrics collection

## Project Structure

```
marxist-chat/
├── src/                # Core source code
│   ├── api/            # API and WebSocket endpoints
│   ├── cli/            # Command-line interface
│   ├── core/           # Core application logic
│   │   ├── feed_processor.py    # RSS feed handling
│   │   ├── llm_manager.py       # LLM initialization and management
│   │   ├── metadata_repository.py # Metadata indexing and retrieval
│   │   ├── query_engine.py      # RAG query processing
│   │   └── vector_store.py      # Vector database operations
│   ├── utils/          # Utility functions
│   ├── app.py          # FastAPI web server entry point
│   ├── config.py       # Configuration management
│   └── main.py         # CLI application entry point
├── static/             # Web interface files
├── kubernetes/         # Kubernetes deployment configs
├── posts_cache/        # Cached RSS articles
├── vector_store/       # Vector database files
├── logs/               # Application logs
└── .env                # Environment configuration
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

### REST API Endpoints

#### Health and Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/healthcheck` | GET | Simple health check |
| `/api/v1/status` | GET | System status with connection counts |
| `/api/v1/metrics` | GET | Detailed system metrics |
| `/api/v1/metrics/summary` | GET | Simplified metrics summary |

#### Content Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/archive-rss` | POST | Download and process RSS feeds |
| `/api/v1/create-vector-store` | POST | Create vector store from archived documents |
| `/api/v1/feed-stats` | GET | Statistics about archived feeds |
| `/api/v1/vector-store-stats` | GET | Statistics about the vector store |

#### Document Access

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/documents/{document_id}` | GET | Get a specific document by ID |
| `/api/v1/documents/search` | GET | Search for documents matching a query |
| `/api/v1/model-info` | GET | Get information about the current LLM model |

#### Query Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Process a query and get response with sources |
| `/api/v1/query/{user_id}/stop` | POST | Stop an in-progress query |

#### Queue Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/queue` | GET | Get detailed queue status |
| `/api/v1/queue/clear` | POST | Clear the waiting queue (admin) |

#### System Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/service/restart` | POST | Restart service components (admin) |
| `/api/v1/rebuild-metadata-index` | POST | Rebuild metadata index from cached documents |

### WebSocket API

WebSocket connections provide real-time chat functionality with token-streaming for responsive interactions.

#### Connection Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v1/ws/chat/{user_id}` | Connect with a specific user ID |
| `/api/v1/ws/chat` | Connect with an auto-generated user ID |

#### Client-to-Server Messages

```json
// Query message
{
  "message": "What is the communist position on healthcare?"
}

// Stop query command
{
  "command": "stop_query"
}
```

#### Server-to-Client Messages

The server sends various message types during WebSocket communication:

1. **System Messages**: Connection status and informational messages
   ```json
   {
     "type": "system",
     "message": "Connected to chat service"
   }
   ```

2. **Queue Messages**: Position updates when in the waiting queue
   ```json
   {
     "type": "queue",
     "position": 3,
     "message": "You are #3 in queue. Estimated wait time: ~6 minutes"
   }
   ```

3. **Stream Messages**: Token-by-token streaming responses
   ```json
   // Stream start
   { "type": "stream_start" }
   
   // Individual tokens
   { "type": "stream_token", "data": "token" }
   
   // Stream end with complete response
   { "type": "stream_end", "data": "The complete response text..." }
   ```

4. **Source Messages**: Citations for the response
   ```json
   {
     "type": "sources",
     "data": [
       {
         "title": "Healthcare as a Human Right",
         "date": "2023-02-15",
         "url": "https://communistusa.org/2023/02/healthcare-human-right",
         "excerpt": "Access to healthcare is a fundamental human right..."
       }
     ]
   }
   ```

5. **Error Messages**: Error notifications
   ```json
   {
     "type": "error",
     "message": "An error occurred: Failed to process query"
   }
   ```

### WebSocket Implementation Example

```javascript
// Connect to WebSocket
const socket = new WebSocket('ws://localhost:8000/api/v1/ws/chat');
let responseText = '';

// Handle messages
socket.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'stream_start':
      responseText = '';
      document.getElementById('response').innerHTML = '';
      document.getElementById('stop-btn').classList.remove('hidden');
      break;
      
    case 'stream_token':
      responseText += data.data;
      document.getElementById('response').innerHTML = responseText;
      break;
      
    case 'stream_end':
      document.getElementById('stop-btn').classList.add('hidden');
      break;
      
    case 'sources':
      // Display sources
      const sourcesEl = document.getElementById('sources');
      sourcesEl.innerHTML = '';
      
      data.data.forEach(source => {
        sourcesEl.innerHTML += `
          <div class="source">
            <h4>${source.title} (${source.date})</h4>
            <p>${source.excerpt}</p>
            <a href="${source.url}" target="_blank">Read more</a>
          </div>
        `;
      });
      break;
      
    case 'error':
      document.getElementById('response').innerHTML += 
        `<div class="error">${data.message}</div>`;
      break;
  }
};

// Send a query
function sendQuery(text) {
  socket.send(JSON.stringify({ message: text }));
}

// Stop a running query
function stopQuery() {
  socket.send(JSON.stringify({ command: "stop_query" }));
}
```

### Error Handling

All API endpoints use standard HTTP status codes:

- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (resource doesn't exist)
- 422: Validation Error (invalid input)
- 500: Internal Server Error

Error responses follow a consistent format:

```json
{
  "status": "error",
  "message": "Error message",
  "details": {} // Optional additional details
}
```

## Web Interface

The application includes a simple web interface for interacting with the chatbot. The interface provides:

- Real-time WebSocket connection to the chatbot
- Token-by-token streaming for responsive chat experience
- Ability to interrupt long-running queries
- Buttons for archiving RSS feeds and creating the vector store
- Status checking capabilities
- Source attribution for responses

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