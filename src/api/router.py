from fastapi import APIRouter, WebSocket, Request, Query, Path
import logging
import uuid
from typing import Optional

from api.websocket import handle_websocket_connection
from api.endpoints import (
    healthcheck, 
    get_status,
    archive_rss,
    create_vector_store,
    process_query,
    get_feed_stats,
    get_vector_store_stats
)

logger = logging.getLogger("api.router")
router = APIRouter()

# Health and status endpoints
@router.get("/healthcheck")
async def api_healthcheck():
    """Health check endpoint"""
    return await healthcheck()

@router.get("/status")
async def api_status():
    """Get system status"""
    return await get_status()

# Archive and vector store management
@router.post("/archive-rss")
async def api_archive_rss():
    """Archive RSS feeds"""
    return await archive_rss()

@router.post("/create-vector-store")
async def api_create_vector_store(overwrite: bool = Query(False)):
    """Create vector store from archived documents"""
    return await create_vector_store(overwrite=overwrite)

# Stats endpoints
@router.get("/feed-stats")
async def api_feed_stats():
    """Get feed statistics"""
    return await get_feed_stats()

@router.get("/vector-store-stats")
async def api_vector_store_stats():
    """Get vector store statistics"""
    return await get_vector_store_stats()

# Query endpoints
@router.post("/query")
async def api_query(request: Request):
    """Process a query"""
    data = await request.json()
    query_text = data.get("query")
    return await process_query(query_text)

# WebSocket endpoint
@router.websocket("/ws/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str = Path(...)):
    """WebSocket endpoint for chat interface with queueing system"""
    await handle_websocket_connection(websocket, user_id)

@router.websocket("/ws/chat")
async def websocket_auto_id(websocket: WebSocket):
    """WebSocket endpoint with auto-generated ID"""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    await handle_websocket_connection(websocket, user_id)