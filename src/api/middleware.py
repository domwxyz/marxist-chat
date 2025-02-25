import time
import logging
import threading
from typing import Dict, List, Any, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import config

logger = logging.getLogger("api.metrics")

class MetricsCollector:
    """Collect and monitor system metrics"""
    
    def __init__(self, interval=30):
        self.interval = interval
        self.running = False
        self.thread = None
        
        # Metrics storage
        self.request_count = 0
        self.request_latencies = []  # Store last 100 request latencies
        self.error_count = 0
        self.memory_usage = []
        self.cpu_usage = []
        self.active_connections = 0
        self.queue_length = 0
        
        # API endpoints metrics
        self.endpoint_stats = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Stats
        self.stats = {}
    
    def start(self):
        """Start the metrics collection thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        logger.info("Metrics collector started")
    
    def stop(self):
        """Stop the metrics collection thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Metrics collector stopped")
    
    def _collect_loop(self):
        """Continuously collect metrics"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Update stats
                self._update_stats()
                
                # Log metrics periodically
                logger.info(f"System metrics: memory={self.stats.get('memory_usage_percent', 'N/A')}%, "
                           f"cpu={self.stats.get('cpu_usage_percent', 'N/A')}%, "
                           f"requests={self.request_count}, "
                           f"connections={self.active_connections}")
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                
            # Sleep for the specified interval
            time.sleep(self.interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            with self.lock:
                self.memory_usage.append(memory.percent)
                if len(self.memory_usage) > 10:
                    self.memory_usage.pop(0)
            
            # CPU usage
            cpu = psutil.cpu_percent(interval=1)
            with self.lock:
                self.cpu_usage.append(cpu)
                if len(self.cpu_usage) > 10:
                    self.cpu_usage.pop(0)
                    
            # Monitor process-specific resources
            process = psutil.Process()
            with self.lock:
                self.stats["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
                self.stats["process_cpu_percent"] = process.cpu_percent(interval=0.1)
                self.stats["process_threads"] = process.num_threads()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _update_stats(self):
        """Update the stats dictionary with current metrics"""
        with self.lock:
            self.stats.update({
                "request_count": self.request_count,
                "error_count": self.error_count,
                "active_connections": self.active_connections,
                "queue_length": self.queue_length,
                "memory_usage_percent": sum(self.memory_usage) / max(len(self.memory_usage), 1) if self.memory_usage else 0,
                "cpu_usage_percent": sum(self.cpu_usage) / max(len(self.cpu_usage), 1) if self.cpu_usage else 0,
                "avg_request_latency_ms": sum(self.request_latencies) / max(len(self.request_latencies), 1) if self.request_latencies else 0,
                "endpoints": self.endpoint_stats,
                "max_concurrent_users": config.MAX_CONCURRENT_USERS,
                "queue_timeout": config.QUEUE_TIMEOUT,
                "request_timeout": config.REQUEST_TIMEOUT
            })
    
    def record_request(self, endpoint: str, latency_ms: float, status_code: int):
        """Record a request with its latency"""
        with self.lock:
            self.request_count += 1
            self.request_latencies.append(latency_ms)
            if len(self.request_latencies) > 100:
                self.request_latencies.pop(0)
                
            # Track per-endpoint stats
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "count": 0,
                    "latencies": [],
                    "errors": 0
                }
            
            self.endpoint_stats[endpoint]["count"] += 1
            self.endpoint_stats[endpoint]["latencies"].append(latency_ms)
            
            # Keep only last 20 latencies per endpoint
            if len(self.endpoint_stats[endpoint]["latencies"]) > 20:
                self.endpoint_stats[endpoint]["latencies"].pop(0)
                
            # Track errors by status code
            if status_code >= 400:
                self.error_count += 1
                self.endpoint_stats[endpoint]["errors"] += 1
    
    def update_connection_stats(self, active: int, queued: int):
        """Update connection statistics"""
        with self.lock:
            self.active_connections = active
            self.queue_length = queued
    
    def get_metrics(self):
        """Get current metrics"""
        with self.lock:
            # Calculate averages for endpoints
            for endpoint, stats in self.endpoint_stats.items():
                if stats["latencies"]:
                    avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
                    self.endpoint_stats[endpoint]["avg_latency_ms"] = avg_latency
            
            return self.stats.copy()  # Return a copy to avoid race conditions

# Singleton instance
metrics_collector = MetricsCollector()

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Initialize response for exception case
        response = None
        status_code = 500
        
        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code
            
            # Get endpoint path - normalize it
            path = request.url.path
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_request(path, latency_ms, status_code)
            
            # Update connection stats regularly
            from api.websocket import get_connection_status
            if request.url.path.endswith("/status") or request.url.path.endswith("/metrics"):
                connection_status = get_connection_status()
                metrics_collector.update_connection_stats(
                    connection_status["active_connections"],
                    connection_status["queue_length"]
                )
            
            return response
            
        except Exception as e:
            # Record error
            latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_request(request.url.path, latency_ms, 500)
            
            # Re-raise the exception
            raise
