import logging
from fastapi import HTTPException, Request
from typing import Dict, List, Any

from core.feed_processor import FeedProcessor
from core.vector_store import VectorStoreManager
from core.query_engine import QueryEngine
from api.websocket import get_connection_status, get_query_engine

logger = logging.getLogger("api.endpoints")

# Singleton instances
_query_engine = None

async def healthcheck() -> Dict[str, str]:
    """Health check endpoint to verify API is running"""
    return {"status": "ok"}

async def get_status() -> Dict[str, Any]:
    """Get system status information"""
    status = get_connection_status()
    status["status"] = "online"
    return status

async def archive_rss() -> Dict[str, Any]:
    """Archive RSS feeds"""
    feed_processor = FeedProcessor()
    try:
        entries = feed_processor.fetch_all_feeds()
        documents = feed_processor.process_entries(entries)
        return {
            "status": "success", 
            "message": f"Successfully archived {len(entries)} RSS feed entries",
            "document_count": len(documents)
        }
    except Exception as e:
        logger.error(f"Error archiving RSS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_vector_store(overwrite: bool = False) -> Dict[str, str]:
    """Create vector store from archived documents"""
    vector_store_manager = VectorStoreManager()
    try:
        index = vector_store_manager.create_vector_store(overwrite=overwrite)
        if index:
            # Reset query engine to use new vector store on next query
            global _query_engine
            _query_engine = None
            return {"status": "success", "message": "Vector store created successfully"}
        else:
            raise HTTPException(
                status_code=400, 
                detail="Failed to create vector store. Check logs for details."
            )
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_query(query_text: str) -> Dict[str, Any]:
    """Process a text query and return the response with sources"""
    try:
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")
            
        query_engine = await get_query_engine()
        response = query_engine.query(query_text)
        sources = query_engine.get_formatted_sources()
        
        return {
            "response": str(response.response),
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_feed_stats() -> Dict[str, Any]:
    """Get statistics about archived feeds"""
    try:
        feed_processor = FeedProcessor()
        
        # Count files in cache
        import config
        from pathlib import Path
        
        cache_dir = config.CACHE_DIR
        file_count = len(list(cache_dir.glob("*.txt"))) if cache_dir.exists() else 0
        
        stats = {
            "feed_count": len(feed_processor.feed_urls),
            "document_count": file_count,
            "feeds": feed_processor.feed_urls
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting feed stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_vector_store_stats() -> Dict[str, Any]:
    """Get statistics about the vector store"""
    try:
        vector_store_manager = VectorStoreManager()
        
        # Check if vector store exists
        import config
        vector_store_exists = config.VECTOR_STORE_DIR.exists()
        
        if not vector_store_exists:
            return {
                "status": "not_found",
                "exists": False
            }
        
        # Load vector store to get stats
        index = vector_store_manager.load_vector_store()
        
        if not index:
            return {
                "status": "error",
                "exists": True,
                "message": "Vector store exists but could not be loaded"
            }
        
        # Get basic stats
        stats = {
            "status": "ok",
            "exists": True,
            "node_count": len(index.docstore.docs) if hasattr(index, 'docstore') else "unknown"
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting vector store stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
