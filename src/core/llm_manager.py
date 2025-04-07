import time
import random
import logging
import functools
import traceback
import requests
import os
from typing import Optional, Callable, Any, Dict

from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen
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
        # Check if we're in distributed mode and using remote LLM service
        if config.DISTRIBUTED_MODE:
            logger.info("Running in distributed mode, using remote LLM service")
            return LLMManager.RemoteLLM(config.LLM_SERVICE_URL)
        
        # If running in local mode, initialize local LLM
        model_url = model_url or config.CURRENT_LLM
        temperature = temperature if temperature is not None else config.TEMPERATURE
        threads = threads or config.NUM_THREADS
        
        logger.info(f"Initializing local LLM model: {model_url}")
        logger.info(f"Parameters: temperature={temperature}, threads={threads}")
        
        # Check if model exists locally
        local_path = None
        if os.path.exists(config.LOCAL_MODEL_PATH):
            logger.info(f"Using cached local model: {config.LOCAL_MODEL_PATH}")
            local_path = str(config.LOCAL_MODEL_PATH)
            model_url = None
        
        try:
            llm = LlamaCPP(
                model_url=model_url,
                model_path=local_path,
                temperature=temperature,
                max_new_tokens=256,
                context_window=4096,
                model_kwargs={"n_threads": threads, 
                              "n_batch": 512, 
                              "use_mlock": True},
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
        if isinstance(model, LLMManager.RemoteLLM):
            return model.is_healthy()
            
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
        if "0.5B" in model_name:
            size = "nano (~1GB)"
        elif "1.5B" in model_name:
            size = "small (~2GB)"
        elif "3B" in model_name:
            size = "medium (~3GB)"
        elif "7B" in model_name:
            size = "large (~5GB)"
        elif "14B" in model_name:
            size = "xlarge (~9GB)"
        
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
            "temperature": config.TEMPERATURE,
            "distributed_mode": config.DISTRIBUTED_MODE
        }
        
    class RemoteLLM(LLM):
        """Wrapper for remote LLM service that properly implements the LLM interface"""
        
        service_url: str  # Define this as a model field
        
        def __init__(self, service_url: str):
            """Initialize the remote LLM with service URL"""
            # Initialize the parent class first
            super().__init__()
            
            # Initialize our fields
            self.service_url = service_url
            self.health_url = f"{service_url}/health"
            self.status_url = f"{service_url}/status"
            self.query_url = f"{service_url}/query"
            logger.info(f"Initialized RemoteLLM with service URL: {service_url}")
        
        def is_healthy(self):
            """Check if the remote LLM service is healthy"""
            try:
                response = requests.get(self.health_url, timeout=5)
                return response.status_code == 200
            except Exception:
                return False

        def get_status(self):
            """Get status information from the remote LLM service"""
            try:
                response = requests.get(self.status_url, timeout=5)
                if response.status_code == 200:
                    return response.json()
                return {"status": "error", "code": response.status_code}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @property
        def metadata(self) -> LLMMetadata:
            """Return metadata about the model"""
            return LLMMetadata(
                model_name="Remote LLM Service",
                model_size="unknown",
                max_input_size=4096,  # Adjust based on your model's capabilities
                context_window=4096,
                is_chat_model=True
            )
        
        def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
            """Complete a prompt using the remote LLM service"""
            try:
                response = requests.post(
                    self.query_url,
                    json={
                        "query_text": prompt,
                        "request_id": f"direct_{int(time.time())}"
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    # Process streaming response to get the complete text
                    full_text = ""
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if "token" in data:
                                full_text += data["token"]
                    
                    # Return a proper CompletionResponse object
                    return CompletionResponse(text=full_text)
                else:
                    raise Exception(f"LLM service error: {response.status_code}")
            except Exception as e:
                logger.error(f"Error in RemoteLLM.complete: {e}")
                raise
        
        def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
            """Streaming completion method"""
            try:
                request_id = f"stream_{int(time.time())}"
                
                response = requests.post(
                    self.query_url,
                    json={
                        "query_text": prompt,
                        "request_id": request_id
                    },
                    stream=True,
                    timeout=300
                )
                
                if response.status_code != 200:
                    raise Exception(f"LLM service error: {response.status_code}")
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if "token" in data:
                            yield data["token"]
            except Exception as e:
                logger.error(f"Error in stream_complete: {e}")
                raise
        
        def chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponse:
            """Implement chat by converting messages to a prompt"""
            prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
            return self.complete(prompt, **kwargs)
        
        def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
            """Implement streaming chat"""
            prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
            yield from self.stream_complete(prompt, **kwargs)
        
        async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
            """Implement async completion by using synchronous version"""
            # This is not truly async but provides compatibility
            return self.complete(prompt, **kwargs)
        
        async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
            """Implement async streaming by using synchronous version"""
            # This is not truly async but provides compatibility
            for token in self.stream_complete(prompt, **kwargs):
                yield token
        
        async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponse:
            """Implement async chat by using synchronous version"""
            return self.chat(messages, **kwargs)
        
        async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
            """Implement async streaming chat by using synchronous version"""
            for token in self.stream_chat(messages, **kwargs):
                yield token
            