import os
import sys
import json
import logging
import time
import asyncio
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "llm_service.log"))
    ]
)
logger = logging.getLogger("llm_service")

# Import your existing core code
try:
    from core.llm_manager import LLMManager
    from core.vector_store import VectorStoreManager
    from core.query_engine import QueryEngine
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    raise

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
llm_instance = None
vector_store_manager = None
query_engine = None
active_requests = {}
request_queue = Queue(maxsize=20)  # Limit queue to prevent memory issues

# Load environment variables
NUM_THREADS = int(os.getenv("NUM_THREADS", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MODEL_URL = os.getenv("CURRENT_LLM", 
                     "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
MODEL_LOCAL_PATH = os.path.join("models", os.path.basename(MODEL_URL))

# Request models
class QueryRequest(BaseModel):
    query_text: str
    request_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class StopRequest(BaseModel):
    request_id: str

async def initialize_services():
    """Initialize the LLM and vector store on startup"""
    global llm_instance, vector_store_manager, query_engine
    
    logger.info("Initializing LLM service...")
    
    # Check if model exists locally
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_LOCAL_PATH):
        logger.info(f"Downloading model from {MODEL_URL} to {MODEL_LOCAL_PATH}")
        # This should be done in the Dockerfile, but we'll add a fallback here
        import requests
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_LOCAL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
    
    # Initialize LLM
    try:
        logger.info(f"Initializing LLM with {NUM_THREADS} threads, temperature {TEMPERATURE}")
        llm_instance = LLMManager.initialize_llm(
            model_url=MODEL_LOCAL_PATH,  # Use local path
            temperature=TEMPERATURE,
            threads=NUM_THREADS
        )
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise
    
    # Initialize vector store
    try:
        logger.info("Initializing vector store manager")
        vector_store_manager = VectorStoreManager()
        logger.info("Vector store manager initialized")
    except Exception as e:
        logger.error(f"Error initializing vector store manager: {e}")
        raise
        
    # Initialize query engine
    try:
        logger.info("Initializing query engine")
        query_engine = QueryEngine()
        success = query_engine.initialize()
        if success:
            logger.info("Query engine initialized successfully")
        else:
            logger.error("Failed to initialize query engine")
            raise Exception("Query engine initialization failed")
    except Exception as e:
        logger.error(f"Error initializing query engine: {e}")
        raise
        
    logger.info("LLM service startup complete")

# Start the request processor thread
def process_requests():
    """Background thread to process requests from the queue"""
    logger.info("Starting request processor thread")
    while True:
        try:
            # Get a request from the queue
            request_id, query_request = request_queue.get()
            logger.info(f"Processing request {request_id}")
            
            if request_id in active_requests and active_requests[request_id].get("stop_event"):
                logger.info(f"Request {request_id} was already stopped")
                request_queue.task_done()
                continue
            
            # Create a stop event for this request
            stop_event = asyncio.Event()
            active_requests[request_id] = {
                "stop_event": stop_event,
                "status": "processing",
                "start_time": time.time()
            }
            
            # Process the query in a non-blocking way
            # The query results are streamed directly to the client
            request_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in request processor: {e}")
            try:
                request_queue.task_done()
            except:
                pass
            time.sleep(1)  # Avoid tight loop if there's an error

# Start the request processor thread on startup
processor_thread = Thread(target=process_requests, daemon=True)
processor_thread.start()

@app.on_event("startup")
async def startup_event():
    await initialize_services()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if query_engine is None:
        raise HTTPException(status_code=503, detail="LLM service initializing")
    return {"status": "healthy"}

@app.post("/query")
async def process_query(query_request: QueryRequest):
    """Process a query and stream the response"""
    request_id = query_request.request_id
    
    # Check if we're already processing this request
    if request_id in active_requests:
        return {"error": "Request already being processed"}
    
    if not query_engine:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    
    logger.info(f"Received query request: {request_id}")
    
    # Return a streaming response
    async def response_stream():
        try:
            # Create a stop event for this request
            stop_event = asyncio.Event()
            active_requests[request_id] = {
                "stop_event": stop_event,
                "status": "processing",
                "start_time": time.time()
            }
            
            # Stream response tokens
            async for token in query_engine.stream_query(
                query_request.query_text,
                stop_event=stop_event,
                start_date=query_request.start_date,
                end_date=query_request.end_date
            ):
                # Check if the query has been stopped
                if stop_event.is_set():
                    yield json.dumps({"status": "stopped"}) + "\n"
                    break
                
                # Return the token
                yield json.dumps({"token": token}) + "\n"
            
            # Get formatted sources
            sources = query_engine.get_formatted_sources()
            yield json.dumps({"sources": sources}) + "\n"
            
            # Mark as completed
            yield json.dumps({"status": "complete"}) + "\n"
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            # Clean up
            if request_id in active_requests:
                del active_requests[request_id]
    
    return StreamingResponse(response_stream(), media_type="text/event-stream")

@app.post("/stop")
async def stop_query(stop_request: StopRequest):
    """Stop a running query"""
    request_id = stop_request.request_id
    
    if request_id in active_requests and "stop_event" in active_requests[request_id]:
        # Set the stop event
        active_requests[request_id]["stop_event"].set()
        logger.info(f"Stopped query {request_id}")
        return {"status": "stopped"}
    
    return {"status": "not_found"}

@app.get("/status")
async def get_status():
    """Get the status of the LLM service"""
    return {
        "active_requests": len(active_requests),
        "queue_length": request_queue.qsize(),
        "llm_initialized": llm_instance is not None,
        "vector_store_initialized": vector_store_manager is not None,
        "query_engine_initialized": query_engine is not None,
    }

if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
    