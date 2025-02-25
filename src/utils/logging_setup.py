import logging
import logging.handlers
import os
from pathlib import Path
import time
import json
from typing import Dict, Any

class JsonFormatter(logging.Formatter):
    """Format logs as JSON for easier parsing and analysis"""
    
    def format(self, record):
        # Create a JSON structure for the log record
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
            
        return json.dumps(log_data)

def setup_logging(log_dir: Path, log_level: str = "INFO", json_format: bool = False):
    """Set up comprehensive logging configuration"""
    # Ensure log directory exists
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    json_formatter = JsonFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(json_formatter if json_format else standard_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler with rotation
    log_file = log_dir / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(json_formatter if json_format else standard_formatter)
    root_logger.addHandler(file_handler)
    
    # Create error-specific log file
    error_file = log_dir / "error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter if json_format else standard_formatter)
    root_logger.addHandler(error_handler)
    
    # Set specific loggers for external libraries to reduce verbosity
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.INFO)
    
    # Log startup information
    logger = logging.getLogger("api")
    logger.info(f"Logging initialized at level {log_level}")
    
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that allows adding request context to logs"""
    
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {'extra_data': {}}
        
        if self.extra:
            kwargs['extra']['extra_data'].update(self.extra)
            
        return msg, kwargs

def get_request_logger(logger, request_id=None, user_id=None):
    """Get a logger with request context information"""
    if not request_id:
        request_id = f"req_{int(time.time() * 1000)}"
        
    extra = {
        "request_id": request_id,
    }
    
    if user_id:
        extra["user_id"] = user_id
        
    return LoggerAdapter(logger, extra)
    