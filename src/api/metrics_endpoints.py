import logging
from fastapi import APIRouter
from typing import Dict, Any

from api.middleware import metrics_collector

logger = logging.getLogger("api.metrics_endpoint")

# Create a router for metrics endpoints
metrics_router = APIRouter(prefix="/metrics", tags=["Metrics"])

@metrics_router.get("/")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    return metrics_collector.get_metrics()

@metrics_router.get("/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """Get a simplified summary of system metrics"""
    full_metrics = metrics_collector.get_metrics()
    
    # Create a simplified summary
    summary = {
        "status": "healthy",
        "requests": {
            "total": full_metrics.get("request_count", 0),
            "errors": full_metrics.get("error_count", 0),
            "avg_latency_ms": full_metrics.get("avg_request_latency_ms", 0)
        },
        "connections": {
            "active": full_metrics.get("active_connections", 0),
            "queued": full_metrics.get("queue_length", 0),
            "max": full_metrics.get("max_concurrent_users", 0)
        },
        "resources": {
            "memory_percent": full_metrics.get("memory_usage_percent", 0),
            "cpu_percent": full_metrics.get("cpu_usage_percent", 0)
        }
    }
    
    # Set status based on resource usage
    if full_metrics.get("memory_usage_percent", 0) > 90 or full_metrics.get("cpu_usage_percent", 0) > 90:
        summary["status"] = "warning"
    
    # Check queue length
    if full_metrics.get("queue_length", 0) > 10:
        summary["status"] = "busy"
    
    return summary

# Function to initialize metrics collector
def start_metrics_collector():
    """Start the metrics collector"""
    metrics_collector.start()
    logger.info("Metrics collector started")
