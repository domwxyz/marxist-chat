from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import atexit

from api.exceptions import setup_exception_handlers 
from api.router import router
from api.middleware import MetricsMiddleware, metrics_collector
from api.metrics_endpoint import metrics_router, start_metrics_collector
from utils.logging_setup import setup_logging
import config

# Configure logging
logger = setup_logging(
    log_dir=config.LOG_DIR,
    log_level=config.LOG_LEVEL,
    json_format=False  # Set to True if you want JSON-formatted logs
)

app = FastAPI(
    title="Marxist Chat API",
    description="API for RAG-based chat interface for communist articles",
    version="1.0.0"
)

# Set up global exception handlers
setup_exception_handlers(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(router, prefix="/api/v1")
app.include_router(metrics_router, prefix="/api/v1")

# Start metrics collector
@app.on_event("startup")
def startup_event():
    """Initialize services on startup"""
    logger.info("Starting API server")
    start_metrics_collector()

# Stop metrics collector on shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server")
    metrics_collector.stop()

# Register shutdown handler
atexit.register(metrics_collector.stop)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )
