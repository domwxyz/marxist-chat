# Marxist Chat API Documentation

This document outlines the API endpoints and WebSocket connections available for integrating with the Marxist Chat system. All examples assume the API is running at `http://localhost:8000`.

## Base URL

All API endpoints are prefixed with `/api/v1`.

## Authentication

The current API implementation does not include authentication. For production deployments, it's recommended to implement authentication for admin endpoints like `/api/v1/queue/clear` and `/api/v1/service/restart`.

## REST API Endpoints

### Health and Status

#### GET `/api/v1/healthcheck`
Simple health check to verify the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

#### GET `/api/v1/status`
Get system status information including connection counts.

**Response:**
```json
{
  "active_connections": 5,
  "queue_length": 2,
  "max_concurrent_users": 30,
  "status": "online"
}
```

#### GET `/api/v1/metrics`
Get detailed system metrics.

**Response:**
```json
{
  "request_count": 1250,
  "error_count": 12,
  "active_connections": 28,
  "queue_length": 5,
  "memory_usage_percent": 68.5,
  "cpu_usage_percent": 42.3,
  "avg_request_latency_ms": 245.6,
  "process_memory_mb": 3245.8,
  "process_cpu_percent": 38.2,
  "process_threads": 8
}
```

#### GET `/api/v1/metrics/summary`
Get a simplified summary of system metrics.

**Response:**
```json
{
  "status": "healthy",
  "requests": {
    "total": 1250,
    "errors": 12,
    "avg_latency_ms": 245.6
  },
  "connections": {
    "active": 28,
    "queued": 5,
    "max": 30
  },
  "resources": {
    "memory_percent": 68.5,
    "cpu_percent": 42.3
  }
}
```

### Content Management

#### POST `/api/v1/archive-rss`
Trigger an RSS feed archiving operation.

**Response:**
```json
{
  "status": "success",
  "message": "Successfully archived 150 RSS feed entries",
  "document_count": 150
}
```

#### POST `/api/v1/create-vector-store`
Create or recreate the vector store from archived documents.

**Query Parameters:**
- `overwrite` (boolean, default: false): Whether to overwrite an existing vector store

**Response:**
```json
{
  "status": "success",
  "message": "Vector store created successfully"
}
```

#### POST `/api/v1/update-vector-store`
Update the vector store with new articles without rebuilding. Compares dates to only add new content.

**Response:**
```json
{
  "status": "success",
  "message": "Vector store updated successfully"
}
```

#### POST `/api/v1/rebuild-metadata-index`
Rebuild the metadata index from cached documents.

**Response:**
```json
{
  "status": "success",
  "message": "Metadata index rebuilt successfully with 3245 entries"
}
```

### Document Access

#### GET `/api/v1/feed-stats`
Get statistics about RSS feeds and archived documents.

**Response:**
```json
{
  "total_feeds": 2,
  "feeds": [
    "https://communistusa.org/feed",
    "https://marxist.com/index.php?format=feed"
  ],
  "feeds_by_directory": {
    "communistusa-org": {
      "count": 1850,
      "feed_url": "https://communistusa.org/feed",
      "path": "/app/posts_cache/communistusa-org"
    },
    "marxist-com": {
      "count": 1350,
      "feed_url": "https://marxist.com/index.php?format=feed",
      "path": "/app/posts_cache/marxist-com"
    }
  },
  "total_documents": 3200
}
```

#### GET `/api/v1/vector-store-stats`
Get statistics about the vector store.

**Response:**
```json
{
  "status": "ok",
  "exists": true,
  "node_count": 12500
}
```

#### GET `/api/v1/documents/{document_id}`
Get a specific document by ID.

**Response:**
```json
{
  "status": "success",
  "document": {
    "id": "2023-05-01_building-worker-power",
    "title": "Building Worker Power Through Unions",
    "date": "2023-05-01",
    "author": "John Smith",
    "url": "https://communistusa.org/2023/05/building-worker-power",
    "text": "Unions are not simply economic organizations but schools of class struggle..."
  }
}
```

#### GET `/api/v1/documents/search`
Search for documents matching a text query.

**Query Parameters:**
- `query` (string, required): Search query text
- `limit` (integer, default: 10): Maximum number of results to return

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "id": "2023-05-01_building-worker-power",
      "title": "Building Worker Power Through Unions",
      "date": "2023-05-01",
      "url": "https://communistusa.org/2023/05/building-worker-power",
      "relevance_score": 0.92,
      "excerpt": "Unions are not simply economic organizations but schools of class struggle..."
    }
  ],
  "count": 5,
  "query": "labor unions"
}
```

#### GET `/api/v1/model-info`
Get information about the current LLM model.

**Response:**
```json
{
  "status": "success",
  "model": {
    "name": "Qwen2.5-3B-Instruct-Q4_K_M",
    "url": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "size": "small (~2GB)",
    "quantization": "4-bit (Q4_K_M)",
    "threads": 4,
    "temperature": 0.2
  }
}
```

### Query Processing

#### POST `/api/v1/query`
Process a text query and return a response with sources.

**Request Body:**
```json
{
  "query": "What is the position on labor unions?"
}
```

**Response:**
```json
{
  "response": "The Revolutionary Communists of America strongly support labor unions as essential organizations for working class power. They advocate for democratic, militant unions that fight for both economic gains and political power...",
  "sources": [
    {
      "title": "Building Worker Power Through Unions",
      "date": "2023-05-01",
      "url": "https://communistusa.org/2023/05/building-worker-power",
      "excerpt": "Unions are not simply economic organizations but schools of class struggle..."
    }
  ]
}
```

#### POST `/api/v1/query/{user_id}/stop`
Stop an in-progress query for a specific user.

**Response:**
```json
{
  "status": "success",
  "message": "Query stopped for user user_abc123"
}
```

### Queue Management

#### GET `/api/v1/queue`
Get detailed queue status and current position information.

**Response:**
```json
{
  "status": "success",
  "queue_length": 8,
  "active_connections": 30,
  "max_concurrent_users": 30,
  "estimated_wait_time": 960,
  "queue_details": [
    {
      "position": 1,
      "user_id": "user_abc123",
      "wait_time": 120,
      "estimated_remaining": 840
    }
  ]
}
```

#### POST `/api/v1/queue/clear`
Admin endpoint to clear the waiting queue.

**Response:**
```json
{
  "status": "success",
  "message": "Cleared 8 connections from the waiting queue"
}
```

### Service Management

#### POST `/api/v1/service/restart`
Admin endpoint to restart the service components.

**Response:**
```json
{
  "status": "success",
  "message": "Service components restarted successfully"
}
```

## WebSocket API

### Chat Connections

#### WebSocket `/api/v1/ws/chat/{user_id}`
Connect to the chat interface with a specific user ID.

#### WebSocket `/api/v1/ws/chat`
Connect to the chat interface with an auto-generated user ID.

### Message Types

#### Client to Server Messages

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

#### Server to Client Messages

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

**Error Messages:**
```json
{
  "type": "error",
  "message": "An error occurred: Failed to process query"
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
      "excerpt": "Access to healthcare is a fundamental human right..."
    }
  ]
}
```

### Streaming Messages

The WebSocket API supports token-by-token streaming for a more responsive user experience:

**Stream Start Message:**
```json
{
  "type": "stream_start"
}
```

**Stream Token Message:**
```json
{
  "type": "stream_token",
  "data": "token"
}
```

**Stream End Message:**
```json
{
  "type": "stream_end",
  "data": "The complete response text..."
}
```

**Query Stopped Message:**
```json
{
  "type": "query_stopped",
  "message": "Query was stopped by user request."
}
```

## Implementation Examples

### WebSocket Streaming Implementation

```javascript
// Connect to WebSocket
const socket = new WebSocket('ws://localhost:8000/api/v1/ws/chat');

// Variables to track streaming state
let isStreaming = false;
let streamingMessage = '';
let responseElement = document.getElementById('response');

socket.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'stream_start':
      isStreaming = true;
      streamingMessage = '';
      responseElement.innerHTML = '';
      document.getElementById('stop-btn').classList.remove('hidden');
      break;
      
    case 'stream_token':
      if (isStreaming) {
        streamingMessage += data.data;
        responseElement.innerHTML = streamingMessage;
        responseElement.scrollTop = responseElement.scrollHeight;
      }
      break;
      
    case 'stream_end':
      isStreaming = false;
      document.getElementById('stop-btn').classList.add('hidden');
      break;
      
    case 'query_stopped':
      isStreaming = false;
      document.getElementById('stop-btn').classList.add('hidden');
      responseElement.innerHTML += '<br><em>Query stopped</em>';
      break;
      
    case 'sources':
      displaySources(data.data);
      break;
  }
};

// Add stop button handler
document.getElementById('stop-btn').addEventListener('click', function() {
  if (isStreaming) {
    socket.send(JSON.stringify({
      command: "stop_query"
    }));
  }
});

// Send a query
function sendQuery(text) {
  socket.send(JSON.stringify({ message: text }));
}

// Display sources in the UI
function displaySources(sources) {
  const sourcesContainer = document.getElementById('sources');
  sourcesContainer.innerHTML = '';
  
  sources.forEach(source => {
    const sourceElement = document.createElement('div');
    sourceElement.className = 'source';
    sourceElement.innerHTML = `
      <h4>${source.title} (${source.date})</h4>
      <p>${source.excerpt}</p>
      <a href="${source.url}" target="_blank">Read more</a>
    `;
    sourcesContainer.appendChild(sourceElement);
  });
}
```

### Document Search Implementation

```javascript
async function searchDocuments(query) {
  try {
    const response = await fetch(`/api/v1/documents/search?query=${encodeURIComponent(query)}&limit=5`);
    const data = await response.json();
    
    if (data.status === 'success') {
      const resultsContainer = document.getElementById('search-results');
      resultsContainer.innerHTML = '';
      
      if (data.results.length === 0) {
        resultsContainer.innerHTML = '<p>No documents found</p>';
        return;
      }
      
      data.results.forEach(doc => {
        const docElement = document.createElement('div');
        docElement.className = 'document-result';
        docElement.innerHTML = `
          <h3><a href="${doc.url}" target="_blank">${doc.title}</a></h3>
          <p class="date">${doc.date}</p>
          <p class="excerpt">${doc.excerpt}</p>
          <p class="score">Relevance: ${(doc.relevance_score * 100).toFixed(1)}%</p>
        `;
        resultsContainer.appendChild(docElement);
      });
    }
  } catch (error) {
    console.error('Error searching documents:', error);
  }
}
```

## Error Handling

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

## Rate Limiting

The system includes implicit rate limiting through the queue system. Once the maximum number of concurrent connections is reached, new users are placed in a queue. This prevents overloading the server while providing a reasonable user experience.
