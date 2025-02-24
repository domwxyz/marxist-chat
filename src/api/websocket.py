import asyncio
import json
import logging
import time
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Tuple, Optional

import config
from core.query_engine import QueryEngine

logger = logging.getLogger("api.websocket")

# Global storage for active connections and waiting queue
active_connections: Dict[str, WebSocket] = {}
waiting_queue: List[Tuple[str, WebSocket, float]] = []  # (user_id, websocket, timestamp)
_query_engine: Optional[QueryEngine] = None

async def get_query_engine() -> QueryEngine:
    """Get or initialize query engine"""
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine()
        success = _query_engine.initialize()
        if not success:
            raise ValueError("Failed to initialize query engine")
    return _query_engine

async def handle_websocket_connection(websocket: WebSocket, user_id: str):
    """Handle WebSocket connection with queueing mechanism"""
    await websocket.accept()
    
    # If under the limit, activate immediately
    if len(active_connections) < config.MAX_CONCURRENT_USERS:
        active_connections[user_id] = websocket
        await websocket.send_json({
            "type": "system", 
            "message": "Connected to chat service"
        })
        logger.info(f"User {user_id} connected")
        await handle_chat(websocket, user_id)
    else:
        # Add to waiting queue with timestamp
        waiting_queue.append((user_id, websocket, time.time()))
        position = len(waiting_queue)
        await websocket.send_json({
            "type": "queue", 
            "position": position,
            "message": f"You are #{position} in queue. Please wait..."
        })
        logger.info(f"User {user_id} added to queue (position {position})")
        await handle_queue(websocket, user_id)

async def handle_chat(websocket: WebSocket, user_id: str):
    """Handle active chat WebSocket interaction"""
    try:
        query_engine = await get_query_engine()
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Acknowledge receipt
            await websocket.send_json({
                "type": "status", 
                "message": "Processing your query..."
            })
            
            # Get query response
            response = query_engine.query(message_data["message"])
            
            # Send the full response
            await websocket.send_json({
                "type": "response",
                "data": str(response.response)
            })
            
            # Send sources
            sources = query_engine.get_formatted_sources()
            await websocket.send_json({
                "type": "sources",
                "data": sources
            })
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket for user {user_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"An error occurred: {str(e)}"
            })
        except:
            pass
    finally:
        # Clean up on disconnect
        if user_id in active_connections:
            del active_connections[user_id]
            
            # Promote next user in queue if any
            await promote_next_in_queue()

async def handle_queue(websocket: WebSocket, user_id: str):
    """Handle WebSocket connection for user in queue"""
    try:
        # Periodically update queue position until connected or disconnected
        while True:
            # Find current position
            position = None
            for idx, (queued_id, _, _) in enumerate(waiting_queue):
                if queued_id == user_id:
                    position = idx + 1
                    break
            
            # If not found in queue, break loop
            if position is None:
                # Check if promoted to active
                if user_id in active_connections:
                    await handle_chat(websocket, user_id)
                break
                
            # Send position update
            await websocket.send_json({
                "type": "queue", 
                "position": position,
                "message": f"You are #{position} in queue. Please wait..."
            })
            
            # Wait before next update
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        # Remove from queue if disconnected
        waiting_queue[:] = [item for item in waiting_queue if item[0] != user_id]
        logger.info(f"User {user_id} left the queue")
        # Update remaining queue positions
        await update_queue_positions()
    except Exception as e:
        logger.error(f"Error in queue handling for user {user_id}: {e}")

async def promote_next_in_queue():
    """Promote the next user in the waiting queue"""
    if not waiting_queue:
        return
        
    # Check for and remove timed-out users
    current_time = time.time()
    timeout_limit = config.QUEUE_TIMEOUT if hasattr(config, 'QUEUE_TIMEOUT') else 120
    waiting_queue[:] = [
        item for item in waiting_queue 
        if current_time - item[2] < timeout_limit
    ]
    
    if not waiting_queue:
        return
        
    # Promote the first user
    user_id, websocket, _ = waiting_queue.pop(0)
    active_connections[user_id] = websocket
    
    try:
        await websocket.send_json({
            "type": "system", 
            "message": "Connected to chat service"
        })
        logger.info(f"Promoted user {user_id} from queue")
        
        # Update queue positions for remaining users
        await update_queue_positions()
    except Exception as e:
        logger.error(f"Error promoting user {user_id}: {e}")
        # Remove from active connections if there was an error
        if user_id in active_connections:
            del active_connections[user_id]

async def update_queue_positions():
    """Update position information for users in queue"""
    for idx, (user_id, websocket, _) in enumerate(waiting_queue):
        position = idx + 1
        try:
            await websocket.send_json({
                "type": "queue", 
                "position": position,
                "message": f"You are #{position} in queue. Please wait..."
            })
        except Exception as e:
            logger.error(f"Error updating queue position for user {user_id}: {e}")

async def broadcast_message(message: Dict):
    """Broadcast a message to all active connections"""
    disconnected = []
    
    for user_id, websocket in active_connections.items():
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to user {user_id}: {e}")
            disconnected.append(user_id)
    
    # Clean up disconnected users
    for user_id in disconnected:
        if user_id in active_connections:
            del active_connections[user_id]

def get_connection_status():
    """Get current connection statistics"""
    return {
        "active_connections": len(active_connections),
        "queue_length": len(waiting_queue),
        "max_concurrent_users": config.MAX_CONCURRENT_USERS
    }
