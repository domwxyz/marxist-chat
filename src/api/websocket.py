import asyncio
import json
import logging
import time
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, List, Tuple, Optional, Any, Set
import traceback

import config
from core.query_engine import QueryEngine

logger = logging.getLogger("api.websocket")

# Global storage for active connections and waiting queue
active_connections: Dict[str, WebSocket] = {}
waiting_queue: List[Tuple[str, WebSocket, float]] = []  # (user_id, websocket, timestamp)
_query_engine: Optional[QueryEngine] = None

active_queries: Dict[str, asyncio.Task] = {}  # Track active query tasks by user_id
query_stop_events: Dict[str, asyncio.Event] = {}  # Events to signal query cancellation

# Keep track of connections being processed to prevent race conditions
processing_connections: Set[str] = set()

# Semaphore to limit concurrent model usage - ties directly to NUM_THREADS
model_semaphore = asyncio.Semaphore(config.NUM_THREADS)

async def get_query_engine() -> QueryEngine:
    """Get or initialize query engine with retry mechanism"""
    global _query_engine
    
    # If engine exists, return it
    if _query_engine is not None:
        return _query_engine
    
    # Initialize with retry
    retries = 3
    for attempt in range(retries):
        try:
            _query_engine = QueryEngine()
            success = _query_engine.initialize()
            if success:
                logger.info("Query engine initialized successfully")
                return _query_engine
            else:
                logger.error(f"Failed to initialize query engine (attempt {attempt+1}/{retries})")
                await asyncio.sleep(2)  # Wait before retry
        except Exception as e:
            logger.error(f"Error initializing query engine (attempt {attempt+1}/{retries}): {e}")
            await asyncio.sleep(2)  # Wait before retry
    
    # If we get here, all attempts failed
    raise ValueError("Failed to initialize query engine after multiple attempts")

async def handle_websocket_connection(websocket: WebSocket, user_id: str):
    """Handle WebSocket connection with queueing mechanism"""
    try:
        await websocket.accept()
        logger.info(f"Connection request from user {user_id}")
        
        # If under the limit and not already being processed, activate immediately
        if len(active_connections) < config.MAX_CONCURRENT_USERS and user_id not in processing_connections:
            processing_connections.add(user_id)
            active_connections[user_id] = websocket
            
            try:
                await websocket.send_json({
                    "type": "system", 
                    "message": "Connected to chat service"
                })
                logger.info(f"User {user_id} connected")
                await handle_chat(websocket, user_id)
            finally:
                if user_id in processing_connections:
                    processing_connections.remove(user_id)
                if user_id in active_connections:
                    del active_connections[user_id]
                    # Promote next user in queue
                    await promote_next_in_queue()
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
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during connection setup for user {user_id}")
        cleanup_user(user_id)
    except Exception as e:
        logger.error(f"Error in connection setup for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
        finally:
            cleanup_user(user_id)

async def handle_chat(websocket: WebSocket, user_id: str):
    """Handle active chat WebSocket interaction with proper async streaming"""
    try:
        # Initialize query engine
        query_engine = await get_query_engine()
        
        # Add timeout for inactive connections
        last_activity_time = time.time()
        max_inactive_time = config.REQUEST_TIMEOUT * 2  # Double the request timeout
        
        while True:
            # Check for inactive timeout
            current_time = time.time()
            if current_time - last_activity_time > max_inactive_time:
                await websocket.send_json({
                    "type": "system",
                    "message": "Connection closed due to inactivity"
                })
                logger.info(f"Closing inactive connection for user {user_id}")
                break
            
            # Receive message with timeout
            try:
                # Set receive timeout to prevent blocked connections
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                last_activity_time = time.time()  # Update last activity time
                message_data = json.loads(data)
            except asyncio.TimeoutError:
                # Just continue the loop, which will check the inactivity timeout
                continue
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid message format. Please send a valid JSON object."
                })
                continue
            
            # Get the message text and filter params
            query_text = message_data.get("message", "").strip()
            command = message_data.get("command", "")
            start_date = message_data.get("start_date", None)
            end_date = message_data.get("end_date", None)

            # Handle stop query command
            if command == "stop_query":
                await stop_query(user_id)
                await websocket.send_json({
                    "type": "system",
                    "message": "Query has been stopped."
                })
                continue

            # Continue with normal query processing if not a command
            if not query_text:
                await websocket.send_json({
                    "type": "error",
                    "message": "Please provide a non-empty query"
                })
                continue
            
            # Acknowledge receipt
            await websocket.send_json({
                "type": "status", 
                "message": "Processing your query..."
            })
            
            # Use semaphore to limit concurrent model usage
            async with model_semaphore:
                try:
                    # Start timing the request
                    start_time = time.time()
                    
                    # Send "stream_start" message to notify frontend that streaming begins
                    await websocket.send_json({
                        "type": "stream_start"
                    })
                    
                    # Create a stop event for this query
                    stop_event = asyncio.Event()
                    query_stop_events[user_id] = stop_event
                    
                    # Process the query with streaming
                    logger.info(f"Processing query with streaming: {query_text[:100]}...")

                    try:
                        # Create a streaming task that uses the stream_query method directly with date filters
                        stream_task = asyncio.create_task(
                            query_engine.stream_query(
                                query_text, 
                                stop_event, 
                                start_date=start_date, 
                                end_date=end_date
                            ).__anext__()
                        )
                        active_queries[user_id] = stream_task
                        
                        # Stream tokens to the client in real-time
                        full_response = ""
                        
                        async for token in query_engine.stream_query(
                            query_text, 
                            stop_event, 
                            start_date=start_date, 
                            end_date=end_date
                        ):
                            # Check if the query was stopped
                            if stop_event.is_set():
                                # Send a message indicating the query was stopped
                                await websocket.send_json({
                                    "type": "query_stopped",
                                    "message": "Query was stopped by user request."
                                })
                                break
                            
                            # Accumulate the message for the stream_end event
                            full_response += token
                            
                            # Send the token immediately to the frontend
                            await websocket.send_json({
                                "type": "stream_token",
                                "data": token
                            })
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Query timeout for user {user_id}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Query processing timed out. Please try a simpler query."
                        })
                        # Clean up resources
                        if user_id in active_queries:
                            del active_queries[user_id]
                        if user_id in query_stop_events:
                            del query_stop_events[user_id]
                        continue
                    except asyncio.CancelledError:
                        logger.info(f"Query cancelled for user {user_id}")
                        await websocket.send_json({
                            "type": "query_stopped",
                            "message": "Query was cancelled."
                        })
                        continue
                    
                    # End timing
                    elapsed_time = time.time() - start_time
                    logger.info(f"Query processed in {elapsed_time:.2f}s for user {user_id}")

                    # Only send stream_end if we didn't stop early
                    if not stop_event.is_set():
                        # Send "stream_end" message to notify frontend that streaming has completed
                        await websocket.send_json({
                            "type": "stream_end",
                            "data": full_response  # This is the full accumulated response
                        })
                        
                        # Store the sources for future retrieval if not already set
                        if not query_engine.last_sources and hasattr(query_engine, 'query_engine'):
                            # Try to extract sources from the query engine directly
                            try:
                                response = query_engine.query_engine.query(query_text)
                                if hasattr(response, 'source_nodes'):
                                    query_engine.last_sources = response.source_nodes
                            except:
                                pass
                        
                        # Get and send sources if available
                        sources = query_engine.get_formatted_sources()
                        await websocket.send_json({
                            "type": "sources",
                            "data": sources
                        })
                    
                    # Clean up resources
                    if user_id in active_queries:
                        del active_queries[user_id]
                    if user_id in query_stop_events:
                        del query_stop_events[user_id]
                    
                except Exception as e:
                    logger.error(f"Error processing query for user {user_id}: {e}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing query: {str(e)}"
                    })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"An error occurred: {str(e)}"
            })
        except:
            pass
    finally:
        # Clean up on disconnect
        cleanup_user(user_id)
        
def cleanup_user(user_id: str):
    """Clean up user connections and processing state"""
    if user_id in active_connections:
        del active_connections[user_id]
    if user_id in processing_connections:
        processing_connections.remove(user_id)
    
    # Remove from waiting queue if present
    global waiting_queue
    waiting_queue = [item for item in waiting_queue if item[0] != user_id]

async def handle_queue(websocket: WebSocket, user_id: str):
    """Handle WebSocket connection for user in queue with improved error handling"""
    try:
        # Track the last time we checked queue position
        last_update_time = time.time()
        update_interval = 5  # seconds between position updates
        
        # Periodically update queue position until connected or disconnected
        while True:
            # Find current position
            position = None
            current_time = time.time()
            
            for idx, (queued_id, _, timestamp) in enumerate(waiting_queue):
                if queued_id == user_id:
                    position = idx + 1
                    
                    # Check for timeout in queue
                    if current_time - timestamp > config.QUEUE_TIMEOUT:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Your session has timed out after waiting too long in queue."
                        })
                        # Remove from queue
                        waiting_queue[:] = [item for item in waiting_queue if item[0] != user_id]
                        await update_queue_positions()
                        return
                    break
            
            # If not found in queue, break loop
            if position is None:
                # Check if promoted to active
                if user_id in active_connections:
                    await handle_chat(websocket, user_id)
                break
            
            # Only send position updates at certain intervals to avoid spamming
            if current_time - last_update_time >= update_interval:
                # Calculate estimated wait time - rough approximation
                estimated_minutes = position * 2  # Assume ~2 minutes per person ahead in queue
                wait_message = f"You are #{position} in queue. "
                
                if estimated_minutes > 0:
                    wait_message += f"Estimated wait time: ~{estimated_minutes} minute"
                    if estimated_minutes != 1:
                        wait_message += "s"
                
                # Send position update
                await websocket.send_json({
                    "type": "queue", 
                    "position": position,
                    "message": wait_message
                })
                
                last_update_time = current_time
            
            # Wait before next check
            await asyncio.sleep(1)
    
    except WebSocketDisconnect:
        # Remove from queue if disconnected
        waiting_queue[:] = [item for item in waiting_queue if item[0] != user_id]
        logger.info(f"User {user_id} left the queue")
        # Update remaining queue positions
        await update_queue_positions()
    except Exception as e:
        logger.error(f"Error in queue handling for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            waiting_queue[:] = [item for item in waiting_queue if item[0] != user_id]
            await websocket.send_json({
                "type": "error",
                "message": f"Queue error: {str(e)}"
            })
        except:
            pass

async def promote_next_in_queue():
    """Promote the next user in the waiting queue with improved error handling"""
    if not waiting_queue:
        return
        
    # Check for and remove timed-out users
    current_time = time.time()
    timeout_limit = config.QUEUE_TIMEOUT
    
    # Remove timed-out users
    timed_out_users = []
    for user_id, websocket, timestamp in waiting_queue:
        if current_time - timestamp >= timeout_limit:
            timed_out_users.append(user_id)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "You have been removed from the queue due to timeout."
                })
            except:
                pass
    
    # Remove timed out users from queue
    if timed_out_users:
        waiting_queue[:] = [
            item for item in waiting_queue 
            if item[0] not in timed_out_users
        ]
    
    if not waiting_queue:
        return
    
    # Only promote if we're under the concurrent user limit
    if len(active_connections) >= config.MAX_CONCURRENT_USERS:
        return
        
    # Get the first user that isn't already being processed
    next_user = None
    for idx, (user_id, websocket, _) in enumerate(waiting_queue):
        if user_id not in processing_connections:
            next_user = (idx, user_id, websocket)
            break
    
    if not next_user:
        return
    
    # Promote the user
    idx, user_id, websocket = next_user
    waiting_queue.pop(idx)
    processing_connections.add(user_id)
    active_connections[user_id] = websocket
    
    try:
        await websocket.send_json({
            "type": "system", 
            "message": "Connected to chat service"
        })
        logger.info(f"Promoted user {user_id} from queue")
        
        # Update queue positions for remaining users
        await update_queue_positions()
        
        # Start chat handling
        await handle_chat(websocket, user_id)
    except Exception as e:
        logger.error(f"Error promoting user {user_id}: {e}")
        logger.error(traceback.format_exc())
        # Remove from active connections if there was an error
        cleanup_user(user_id)

async def update_queue_positions():
    """Update position information for users in queue"""
    for idx, (user_id, websocket, _) in enumerate(waiting_queue):
        position = idx + 1
        try:
            # Calculate estimated wait time - rough approximation
            estimated_minutes = position * 2  # Assume ~2 minutes per person ahead in queue
            wait_message = f"You are #{position} in queue. "
            
            if estimated_minutes > 0:
                wait_message += f"Estimated wait time: ~{estimated_minutes} minute"
                if estimated_minutes != 1:
                    wait_message += "s"
            
            await websocket.send_json({
                "type": "queue", 
                "position": position,
                "message": wait_message
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
        cleanup_user(user_id)

def get_connection_status():
    """Get current connection statistics"""
    return {
        "active_connections": len(active_connections),
        "queue_length": len(waiting_queue),
        "max_concurrent_users": config.MAX_CONCURRENT_USERS
    }
    
async def reset_query_engine():
    """Reset the query engine to reload models and vector store"""
    global _query_engine
    _query_engine = None
    logger.info("Query engine reset - will be reinitialized on next request")
    return True

async def clear_waiting_queue():
    """Clear the waiting queue and notify users"""
    count = len(waiting_queue)
    
    # Notify users being removed from queue
    for user_id, websocket, _ in waiting_queue:
        try:
            await websocket.send_json({
                "type": "system",
                "message": "You have been removed from the queue due to an administrative action."
            })
        except Exception:
            # Ignore errors for disconnected sockets
            pass
    
    # Clear the queue
    waiting_queue.clear()
    logger.info(f"Cleared {count} connections from waiting queue")
    return count

def get_queue_status():
    """Get detailed queue status information"""
    # Get basic connection stats
    stats = get_connection_status()
    
    # Calculate estimated wait times
    queue_length = stats["queue_length"]
    estimated_wait_time = 0
    
    if queue_length > 0:
        # Assuming each user takes about 2 minutes on average
        estimated_wait_time = queue_length * 2 * 60  # in seconds
    
    stats["estimated_wait_time"] = estimated_wait_time
    
    # Add queue position information
    queue_positions = []
    current_time = time.time()
    
    for idx, (user_id, _, timestamp) in enumerate(waiting_queue):
        wait_time = current_time - timestamp
        estimated_remaining = estimated_wait_time - wait_time if wait_time < estimated_wait_time else 0
        
        queue_positions.append({
            "position": idx + 1,
            "user_id": user_id,
            "wait_time": int(wait_time),
            "estimated_remaining": int(estimated_remaining)
        })
    
    stats["queue_details"] = queue_positions
    return stats

async def stop_query(user_id: str) -> bool:
    """Stop an in-progress query for a user"""
    if user_id not in query_stop_events:
        logger.warning(f"No active query found for user {user_id}")
        return False
    
    # Set the stop event to signal cancellation
    query_stop_events[user_id].set()
    
    # Try to cancel the task if it exists
    if user_id in active_queries and not active_queries[user_id].done():
        # Give a short grace period for natural cancellation
        try:
            await asyncio.wait_for(active_queries[user_id], timeout=1.0)
        except asyncio.TimeoutError:
            # Force cancel if it doesn't stop naturally
            active_queries[user_id].cancel()
        except Exception:
            # Ignore other exceptions
            pass
    
    # Clean up
    if user_id in active_queries:
        del active_queries[user_id]
    if user_id in query_stop_events:
        del query_stop_events[user_id]
    
    logger.info(f"Query stopped for user {user_id}")
    return True
