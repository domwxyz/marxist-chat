import time
import random
import logging
import functools
import traceback
from typing import Optional, Callable, Any

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import config

logger = logging.getLogger("core.llm_manager")

def retry_with_exponential_backoff(
    max_retries: int = 3, 
    initial_delay: float = 1.0, 
    exponential_base: float = 2.0, 
    jitter: bool = True,
    logger_name: str = None
):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the function name for logging
            func_name = func.__name__
            
            # Set up logger
            log = logger
            if logger_name:
                log = logging.getLogger(logger_name)
            
            # Initialize retry state
            num_retries = 0
            delay = initial_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    num_retries += 1
                    
                    if num_retries > max_retries:
                        log.error(f"Failed after {max_retries} retries: {func_name}")
                        log.error(traceback.format_exc())
                        raise
                    
                    # Add jitter to avoid thundering herd problem
                    sleep_time = delay * (exponential_base ** (num_retries - 1))
                    if jitter:
                        sleep_time = sleep_time * (1 + random.random() * 0.1)
                    
                    log.warning(
                        f"Retry {num_retries}/{max_retries} for {func_name} "
                        f"after {sleep_time:.2f}s due to: {str(e)}"
                    )
                    time.sleep(sleep_time)
        return wrapper
    return decorator


class LLMManager:
    """Manager class for LLM models with improved error handling and retries"""
    
    @staticmethod
    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    def initialize_llm(model_url=None, temperature=None, threads=None):
        """Initialize LLM with specified parameters or use defaults from config with retry logic"""
        model_url = model_url or config.CURRENT_LLM
        temperature = temperature if temperature is not None else config.TEMPERATURE
        threads = threads or config.NUM_THREADS
        
        logger.info(f"Initializing LLM model: {model_url}")
        logger.info(f"Parameters: temperature={temperature}, threads={threads}")
        
        try:
            llm = LlamaCPP(
                model_url=model_url,
                model_path=None,
                temperature=temperature,
                max_new_tokens=256,
                context_window=8192,
                model_kwargs={"n_threads": threads},
                verbose=False
            )
            
            logger.info("LLM initialization successful")
            return llm
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @staticmethod
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def initialize_embedding_model(model_name=None):
        """Initialize the embedding model with retry mechanism"""
        model_name = model_name or config.CURRENT_EMBED
        
        logger.info(f"Initializing embedding model: {model_name}")
        
        try:
            embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                max_length=512,
                embed_batch_size=32
            )
            
            logger.info("Embedding model initialization successful")
            return embed_model
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @staticmethod
    def is_model_loaded(model):
        """Check if a model is loaded and functioning"""
        try:
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                return True
            return False
        except:
            return False
    
    @staticmethod
    def get_model_info(model_url=None):
        """Get information about a model"""
        model_url = model_url or config.CURRENT_LLM
        
        # Parse the model URL to extract information
        model_name = model_url.split('/')[-1].split('.')[0]
        
        # Determine size category based on model name
        size = "unknown"
        if "3B" in model_name:
            size = "small (~2GB)"
        elif "7B" in model_name:
            size = "medium (~5GB)"
        elif "14B" in model_name:
            size = "large (~9GB)"
        
        # Get quantization info
        quantization = "unknown"
        if "Q4_K_M" in model_name:
            quantization = "4-bit (Q4_K_M)"
        
        return {
            "name": model_name,
            "url": model_url,
            "size": size,
            "quantization": quantization,
            "threads": config.NUM_THREADS,
            "temperature": config.TEMPERATURE
        }