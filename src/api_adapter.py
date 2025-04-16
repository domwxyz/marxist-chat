import os
import sys
import json
import logging
import traceback
import time
import aiohttp
import asyncio
import uuid
import requests
from typing import Dict, List, Any, Optional

import redis
from redis.asyncio import Redis as AsyncRedis  # Updated import for async Redis
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "api_service.log"))
    ]
)
logger = logging.getLogger("api_service")

# Import your existing API modules
try:
    from api.router import router
    from api.exceptions import setup_exception_handlers
    from api.middleware import MetricsMiddleware, metrics_collector
    from api.metrics_endpoints import metrics_router, start_metrics_collector
    from utils.logging_setup import setup_logging
except ImportError as e:
    logger.error(f"Failed to import API modules: {e}")
    raise

# Load environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm:5000")
API_ID = os.getenv("API_ID", "1")  # Unique identifier for this API instance
PORT = int(os.getenv("PORT", "8000"))

# Global variables
redis_client = None
aioredis_client = None
active_connections = {}  # Store active WebSocket connections
queue_data = {}  # Store queue position information

# Initialize FastAPI app
app = FastAPI(
    title="Marxist Chat API",
    description="API for RAG-based chat interface for communist articles",
    version="1.0.0"
)

# Set up global exception handlers
setup_exception_handlers(app)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# Include existing routers - only include the endpoints we need
app.include_router(router, prefix="/api/v1")
app.include_router(metrics_router, prefix="/api/v1")

async def initialize_redis():
    """Initialize Redis connections"""
    global redis_client, aioredis_client
    
    logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    # Standard Redis client for synchronous operations
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    # Updated: Use the new aioredis API
    aioredis_client = AsyncRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        encoding="utf-8"
    )
    
    # Register this API instance
    instance_key = f"api:instance:{API_ID}"
    redis_client.set(instance_key, json.dumps({
        "started_at": time.time(),
        "host": os.getenv("HOSTNAME", "unknown")
    }))
    redis_client.expire(instance_key, 60)  # 60 second TTL, will be refreshed by health check
    
    logger.info("Redis initialized successfully")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting API service...")
    await initialize_redis()
    start_metrics_collector()
    logger.info(f"API service {API_ID} started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API service...")
    metrics_collector.stop()
    
    # Close Redis connections
    if aioredis_client:
        await aioredis_client.close()
    
    if redis_client:
        redis_client.delete(f"api:instance:{API_ID}")
        redis_client.close()
    
    logger.info("API service shutdown complete")

async def health_check_loop():
    """Run periodic health checks and update Redis TTL"""
    while True:
        try:
            # Update this instance's TTL
            instance_key = f"api:instance:{API_ID}"
            redis_client.set(instance_key, json.dumps({
                "started_at": time.time(),
                "host": os.getenv("HOSTNAME", "unknown"),
                "connections": len(active_connections),
                "updated_at": time.time()
            }))
            redis_client.expire(instance_key, 60)
            
            # Check LLM service health
            try:
                response = requests.get(f"{LLM_SERVICE_URL}/health", timeout=5)
                llm_healthy = response.status_code == 200
            except:
                llm_healthy = False
            
            # Update LLM service status
            redis_client.set(f"llm:status:{API_ID}", "healthy" if llm_healthy else "unhealthy")
            redis_client.expire(f"llm:status:{API_ID}", 60)
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
        
        await asyncio.sleep(15)

# Start health check loop
@app.on_event("startup")
async def start_health_checks():
    asyncio.create_task(health_check_loop())

@app.websocket("/api/v1/ws/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Handle WebSocket chat connections with LLM service forwarding"""
    # Extract WebSocket handling logic from your existing code
    # Implement request forwarding to LLM service
    await handle_websocket(websocket, user_id)

@app.websocket("/api/v1/ws/chat")
async def websocket_auto_id(websocket: WebSocket):
    """WebSocket endpoint with auto-generated ID"""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    await handle_websocket(websocket, user_id)

async def handle_websocket(websocket: WebSocket, user_id: str):
    """Handle WebSocket connection with queueing mechanism"""
    try:
        await websocket.accept()
        logger.info(f"Connection request from user {user_id}")
        
        # Store in active connections
        active_connections[user_id] = websocket
        
        # Send welcome message
        await websocket.send_json({
            "type": "system", 
            "message": f"Connected to chat service (API {API_ID})"
        })
        
        # Handle the chat connection
        await handle_chat(websocket, user_id)
    
    except Exception as e:
        logger.error(f"Error handling WebSocket for {user_id}: {e}")
        # Clean up connection
        if user_id in active_connections:
            del active_connections[user_id]

async def handle_chat(websocket: WebSocket, user_id: str):
    """Handle the chat session and forward requests to LLM service"""
    try:
        # Create a unique client ID for this connection
        client_id = f"{user_id}_{uuid.uuid4().hex[:6]}"
        
        # Process messages from client
        while True:
            # Wait for a message
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
            except Exception as e:
                logger.error(f"Error receiving message from {user_id}: {e}")
                break
            
            # Get the message or command
            if "command" in message_data and message_data["command"] == "stop_query":
                # Stop the query
                request_id = queue_data.get(user_id, {}).get("request_id")
                
                if request_id:
                    try:
                        # Send stop request to LLM service
                        async with aiohttp.ClientSession() as session:
                            await session.post(
                                f"{LLM_SERVICE_URL}/stop",
                                json={"request_id": request_id},
                                timeout=5
                            )
                        
                        # Notify client
                        await websocket.send_json({
                            "type": "query_stopped",
                            "message": "Query was stopped by user request."
                        })
                        
                        # Clean up queue data
                        if user_id in queue_data:
                            del queue_data[user_id]
                            
                    except Exception as e:
                        logger.error(f"Error stopping query for {user_id}: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error stopping query: {str(e)}"
                        })
                continue
            
            # Process a normal message query
            query_text = message_data.get("message", "").strip()
            if not query_text:
                await websocket.send_json({
                    "type": "error",
                    "message": "Please provide a non-empty query"
                })
                continue
            
            # Tell client we're processing
            await websocket.send_json({
                "type": "status", 
                "message": "Processing your query..."
            })
            
            # Create a unique request ID
            request_id = f"req_{uuid.uuid4().hex}"
            
            # Store request info
            queue_data[user_id] = {
                "request_id": request_id,
                "start_time": time.time(),
                "query": query_text
            }
            
            # Send "stream_start" message to begin streaming
            await websocket.send_json({
                "type": "stream_start"
            })
            
            # Send request to LLM service using aiohttp
            try:
                # Create the streaming request to the LLM service using aiohttp
                async with aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(
                        limit=10,  # Increase connection pool
                        keepalive_timeout=60.0  # Keep connections warm
                    )
                ) as session:
                    try:
                        async with session.post(
                            f"{LLM_SERVICE_URL}/query",
                            json={
                                "query_text": query_text, 
                                "request_id": request_id,
                                "start_date": message_data.get("start_date"),
                                "end_date": message_data.get("end_date")
                            },
                            timeout=aiohttp.ClientTimeout(total=600)  # 10 minute timeout
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(f"LLM service error: {response.status}, {error_text}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"LLM service error: {response.status}"
                                })
                                continue
                            
                            # Stream the response
                            full_response = ""
                            sources = None
                            
                            # Process the streaming response line by line
                            async for line in response.content:
                                line_text = line.decode('utf-8').strip()
                                if not line_text:
                                    continue
                                
                                try:
                                    # Parse the JSON line
                                    data = json.loads(line_text)
                                    
                                    # Process the different response types
                                    if "token" in data:
                                        token = data["token"]
                                        full_response += token
                                        
                                        # Forward to client
                                        await websocket.send_json({
                                            "type": "stream_token",
                                            "data": token
                                        })
                                        
                                    elif "sources" in data:
                                        sources = data["sources"]
                                        
                                        # Forward sources to client
                                        await websocket.send_json({
                                            "type": "sources",
                                            "data": sources
                                        })
                                        
                                    elif "status" in data:
                                        if data["status"] == "complete":
                                            # Query complete
                                            await websocket.send_json({
                                                "type": "stream_end",
                                                "data": full_response
                                            })
                                            
                                        elif data["status"] == "stopped":
                                            # Query was stopped
                                            await websocket.send_json({
                                                "type": "query_stopped",
                                                "message": "Query was stopped by user request."
                                            })
                                    
                                    elif "error" in data:
                                        # Handle error
                                        await websocket.send_json({
                                            "type": "error",
                                            "message": data["error"]
                                        })
                                
                                except json.JSONDecodeError:
                                    logger.error(f"Invalid JSON in LLM response: {line_text}")
                                except Exception as e:
                                    logger.error(f"Error processing response line: {str(e)}")
                                    logger.error(traceback.format_exc())
                    except aiohttp.ClientError as e:
                        logger.error(f"AIOHTTP client error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error communicating with LLM service: {str(e)}"
                        })
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout communicating with LLM service")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Request timed out"
                        })
            except Exception as e:
                logger.error(f"Unhandled error in handle_chat: {e}")
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unexpected error: {str(e)}"
                })
            
            # Clean up queue data
            if user_id in queue_data:
                del queue_data[user_id]
    
    except Exception as e:
        logger.error(f"Error in chat handler for {user_id}: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up when the connection closes
        if user_id in active_connections:
            del active_connections[user_id]
        if user_id in queue_data:
            del queue_data[user_id]

@app.get("/api/v1/api-status")
async def get_api_status():
    """Get the status of this API instance"""
    return {
        "api_id": API_ID,
        "active_connections": len(active_connections),
        "connected_to_redis": redis_client is not None and redis_client.ping(),
        "llm_service_url": LLM_SERVICE_URL,
        "uptime_seconds": time.time() - startup_time
    }

# Track startup time
startup_time = time.time()

# Serve the static files from the static directory
@app.get("/")
async def serve_spa():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
    