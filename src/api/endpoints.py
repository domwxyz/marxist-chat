import logging
from fastapi import HTTPException, Request
from typing import Dict, List, Any, Optional

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

async def process_query(query_text: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """Process a text query with optional date filters and return the response with sources"""
    try:
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")
            
        query_engine = await get_query_engine()
        response = query_engine.query(query_text, start_date=start_date, end_date=end_date)
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
        
        # Get organized stats by feed directory
        stats = {
            "total_feeds": len(feed_processor.feed_urls),
            "feeds": feed_processor.feed_urls,
            "feeds_by_directory": {}
        }
        
        # Get the base cache directory
        cache_dir = config.CACHE_DIR
        
        # Count files in the main cache directory (old structure)
        if cache_dir.exists():
            main_dir_files = len(list(cache_dir.glob("*.txt")))
            if main_dir_files > 0:
                stats["feeds_by_directory"]["main_directory"] = {
                    "count": main_dir_files,
                    "path": str(cache_dir)
                }
        
        # Count files in each feed subdirectory (new structure)
        total_documents = 0
        for feed_url in feed_processor.feed_urls:
            feed_dir_name = feed_processor._get_feed_directory_name(feed_url)
            feed_dir = cache_dir / feed_dir_name
            
            if feed_dir.exists():
                file_count = len(list(feed_dir.glob("*.txt")))
                total_documents += file_count
                
                stats["feeds_by_directory"][feed_dir_name] = {
                    "count": file_count,
                    "feed_url": feed_url,
                    "path": str(feed_dir)
                }
        
        # Add total count
        stats["total_documents"] = total_documents
        
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

async def get_queue_status() -> Dict[str, Any]:
    """Get detailed information about the current queue status"""
    from api.websocket import get_queue_status as ws_queue_status
    
    queue_info = ws_queue_status()
    return {
        "status": "success",
        "queue_length": queue_info["queue_length"],
        "active_connections": queue_info["active_connections"],
        "max_concurrent_users": queue_info["max_concurrent_users"],
        "estimated_wait_time": queue_info.get("estimated_wait_time", 0)
    }

async def clear_queue() -> Dict[str, Any]:
    """Admin endpoint to clear the waiting queue"""
    from api.websocket import clear_waiting_queue
    
    cleared_count = await clear_waiting_queue()
    return {
        "status": "success",
        "message": f"Cleared {cleared_count} connections from the waiting queue"
    }

async def get_document_by_id(document_id: str) -> Dict[str, Any]:
    """Get a specific document by ID"""
    # This requires implementing a document ID system in your vector store
    try:
        from core.vector_store import VectorStoreManager
        
        vector_store = VectorStoreManager()
        document = vector_store.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
            
        return {
            "status": "success",
            "document": {
                "id": document_id,
                "title": document.metadata.get("title", "Untitled"),
                "date": document.metadata.get("date", "Unknown"),
                "author": document.metadata.get("author", "Unknown"),
                "url": document.metadata.get("url", ""),
                "text": document.text[:1000] + "..." if len(document.text) > 1000 else document.text
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def search_documents(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search for documents matching a text query"""
    try:
        from core.vector_store import VectorStoreManager
        
        vector_store = VectorStoreManager()
        results = vector_store.search_documents(query, limit=limit)
        
        documents = []
        for doc, score in results:
            documents.append({
                "id": doc.metadata.get("file_name", "").replace(".txt", ""),
                "title": doc.metadata.get("title", "Untitled"),
                "date": doc.metadata.get("date", "Unknown"),
                "url": doc.metadata.get("url", ""),
                "relevance_score": float(score),
                "excerpt": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            })
            
        return {
            "status": "success",
            "results": documents,
            "count": len(documents),
            "query": query
        }
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_model_info() -> Dict[str, Any]:
    """Get information about the current LLM model"""
    try:
        from core.llm_manager import LLMManager
        
        model_info = LLMManager.get_model_info()
        return {
            "status": "success",
            "model": model_info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def restart_service() -> Dict[str, str]:
    """Admin endpoint to restart the service components"""
    try:
        # Reset the query engine
        from api.websocket import reset_query_engine
        await reset_query_engine()
        
        return {
            "status": "success",
            "message": "Service components restarted successfully"
        }
    except Exception as e:
        logger.error(f"Error restarting service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def stop_user_query(user_id: str) -> Dict[str, str]:
    """Stop an in-progress query for a user"""
    try:
        from api.websocket import stop_query
        success = await stop_query(user_id)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail=f"No active query found for user {user_id}"
            )
            
        return {
            "status": "success",
            "message": f"Query stopped for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Error stopping query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def rebuild_metadata_index():
    """Rebuild the metadata index from cached documents"""
    try:
        from core.metadata_repository import MetadataRepository
        import config
        
        metadata_repo = MetadataRepository(config.CACHE_DIR)
        success = metadata_repo.build_metadata_index(force_rebuild=True)
        
        if success:
            return {
                "status": "success",
                "message": f"Metadata index rebuilt successfully with {len(metadata_repo.metadata_list)} entries"
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to rebuild metadata index"
            )
    except Exception as e:
        logger.error(f"Error rebuilding metadata index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
async def update_vector_store() -> Dict[str, Any]:
    """Update vector store with new documents from RSS feed"""
    vector_store_manager = VectorStoreManager()
    try:
        success = vector_store_manager.update_vector_store()
        if success:
            # Reset query engine to use updated vector store on next query
            global _query_engine
            _query_engine = None
            return {"status": "success", "message": "Vector store updated successfully"}
        else:
            raise HTTPException(
                status_code=400, 
                detail="Failed to update vector store. Check logs for details."
            )
    except Exception as e:
        logger.error(f"Error updating vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))