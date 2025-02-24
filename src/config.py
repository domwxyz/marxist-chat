from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent  # Get the parent directory
CACHE_DIR = PARENT_DIR / "posts_cache"  # One level up from BASE_DIR (src)
VECTOR_STORE_DIR = PARENT_DIR / "vector_store"  # One level up from BASE_DIR (src)

# RSS feed configuration
RSS_FEED_URLS = [
    "https://communistusa.org/feed",
    # Add more feeds here
]

# Embedding models
BGE_M3 = "BAAI/bge-m3"
GTE_SMALL = "thenlper/gte-small"
CURRENT_EMBED = BGE_M3  # Default embedding model

# LLM models - listed smallest to largest (2GB-5GB-9GB in download size)
QWEN2_5_3B = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
QWEN2_5_7B = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/resolve/main/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
QWEN2_5_14B = "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-1M-GGUF/resolve/main/Qwen2.5-14B-Instruct-1M-Q4_K_M.gguf"
CURRENT_LLM = QWEN2_5_3B  # Default LLM model

# LLM configuration
NUM_THREADS = 4  # Default thread count for LLM inference
TEMPERATURE = 0.2  # Default temperature for responses

# System prompt for LLM
SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

# API Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8000  # Default port
DEBUG = False  # Debug mode
MAX_CONCURRENT_USERS = 20  # Maximum number of concurrent WebSocket connections
QUEUE_TIMEOUT = 300  # Seconds a user can stay in queue before timing out (5 minutes)
REQUEST_TIMEOUT = 120  # Seconds to wait for a response before timing out (2 minutes)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "logs" / "app.log"
