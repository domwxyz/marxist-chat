from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import config

class LLMManager:
    @staticmethod
    def initialize_llm(model_url=None, temperature=None, threads=None):
        """Initialize LLM with specified parameters or use defaults from config"""
        model_url = model_url or config.CURRENT_LLM
        temperature = temperature if temperature is not None else config.TEMPERATURE
        threads = threads or config.NUM_THREADS
        
        return LlamaCPP(
            model_url=model_url,
            model_path=None,
            temperature=temperature,
            max_new_tokens=1024,
            context_window=8192,
            model_kwargs={"n_threads": threads},
            verbose=False,
            streaming=True  # Enable streaming for future WebSocket implementation
        )
    
    @staticmethod
    def initialize_embedding_model(model_name=None):
        """Initialize the embedding model"""
        model_name = model_name or config.CURRENT_EMBED
        
        return HuggingFaceEmbedding(
            model_name=model_name,
            max_length=512,
            embed_batch_size=64
        )
