# Enhanced API router implementation
from fastapi import APIRouter, WebSocket, Request, Query, Path, Depends, HTTPException
import logging
import uuid
from typing import Optional, List

from api.websocket import handle_websocket_connection
from api.endpoints import (
    healthcheck, 
    get_status,
    archive_rss,
    create_vector_store,
    process_query,
    get_feed_stats,
    get_vector_store_stats,
    # New endpoint functions to add:
    get_queue_status,
    clear_queue,
    get_document_by_id,
    search_documents,
    get_model_info,
    restart_service
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

# Queue management endpoints
@router.get("/queue")
async def api_queue_status():
    """Get queue status and current position information"""
    return await get_queue_status()

@router.post("/queue/clear")
async def api_clear_queue():
    """Admin endpoint to clear the waiting queue"""
    return await clear_queue()

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

# Document access endpoints
@router.get("/documents/{document_id}")
async def api_get_document(document_id: str):
    """Get a specific document by ID"""
    return await get_document_by_id(document_id)

@router.get("/documents/search")
async def api_search_documents(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100)
):
    """Search for documents matching a text query"""
    return await search_documents(query, limit)

# Model info endpoint
@router.get("/model-info")
async def api_model_info():
    """Get information about the current LLM model"""
    return await get_model_info()

# Query endpoints
@router.post("/query")
async def api_query(request: Request):
    """Process a query"""
    data = await request.json()
    query_text = data.get("query")
    return await process_query(query_text)

# Service management
@router.post("/service/restart")
async def api_restart_service():
    """Admin endpoint to restart the service components"""
    return await restart_service()

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
    
@router.post("/query/{user_id}/stop")
async def api_stop_query(user_id: str):
    """Stop an in-progress query for a user"""
    from api.websocket import stop_query
    success = await stop_query(user_id)
    
    if success:
        return {"status": "success", "message": f"Query stopped for user {user_id}"}
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"No active query found for user {user_id}"
        )
