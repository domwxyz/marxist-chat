# Marxist Chat API Documentation

This document outlines the API endpoints and WebSocket connections available for frontend integration with the Marxist Chat system.

## Base URL

All API endpoints are prefixed with `/api/v1`.

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

### RSS and Vector Store Management

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

### Statistics

#### GET `/api/v1/feed-stats`
Get statistics about RSS feeds and archived documents.

**Response:**
```json
{
  "feed_count": 3,
  "document_count": 3200,
  "feeds": ["https://communistusa.org/feed", "..."]
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

### Query

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
    },
    ...
  ]
}
```

## WebSocket API

### Chat Connection

#### WebSocket `/api/v1/ws/chat/{user_id}`
Connect to the chat interface with a specific user ID.

#### WebSocket `/api/v1/ws/chat`
Connect to the chat interface with an auto-generated user ID.

### WebSocket Message Formats

#### Client to Server
```json
{
  "message": "What is the communist position on healthcare?"
}
```

#### Server to Client

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
  "message": "You are #3 in queue. Please wait..."
}
```

**Status Messages:**
```json
{
  "type": "status",
  "message": "Processing your query..."
}
```

**Response Messages:**
```json
{
  "type": "response",
  "data": "The communist position on healthcare is that it should be universal, free at the point of service, and democratically controlled..."
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
    },
    ...
  ]
}
```

**Error Messages:**
```json
{
  "type": "error",
  "message": "An error occurred: Failed to process query"
}
```

### Streaming WebSocket Messages

The WebSocket API supports token-by-token streaming for a more responsive user experience. The streaming process consists of the following message types:

**Stream Start Message:**
Indicates the beginning of a streaming response.
```json
{
  "type": "stream_start"
}
```

**Stream Token Message:**
Contains a single token of the streaming response. These messages will be sent continuously as tokens are generated.
```json
{
  "type": "stream_token",
  "data": "token"
}
```

**Stream End Message:**
Indicates the end of a streaming response and contains the complete response.
```json
{
  "type": "stream_end",
  "data": "The complete response text..."
}
```

After the streaming is complete, the system will send a "sources" message with relevant source information.

### Frontend Streaming Implementation Example

```javascript
// Connect to WebSocket as usual
const socket = new WebSocket('ws://your-server.com/api/v1/ws/chat');

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
      break;
      
    case 'stream_token':
      if (isStreaming) {
        streamingMessage += data.data;
        responseElement.innerHTML = streamingMessage;
        // Auto-scroll to show new content
        responseElement.scrollTop = responseElement.scrollHeight;
      }
      break;
      
    case 'stream_end':
      isStreaming = false;
      // Final response is in data.data if needed
      break;
      
    case 'sources':
      // Display sources as before
      displaySources(data.data);
      break;
      
    // Handle other message types as before
  }
};
```
