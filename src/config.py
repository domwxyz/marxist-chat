import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent  # Get the parent directory
CACHE_DIR = Path(os.getenv("CACHE_DIR", PARENT_DIR / "posts_cache"))  # One level up from BASE_DIR (src)
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", PARENT_DIR / "vector_store"))  # One level up from BASE_DIR (src)
LOG_DIR = Path(os.getenv("LOG_DIR", PARENT_DIR / "logs"))
LOG_DIR.mkdir(exist_ok=True, parents=True)

# RSS feed configuration
DEFAULT_RSS_FEEDS = [
    "https://communistusa.org/feed",
    # Add more feeds here
]
RSS_FEED_URLS = os.getenv("RSS_FEED_URLS", "").split(",") if os.getenv("RSS_FEED_URLS") else DEFAULT_RSS_FEEDS

# Embedding models
BGE_M3 = "BAAI/bge-m3"
GTE_SMALL = "thenlper/gte-small"
CURRENT_EMBED = os.getenv("CURRENT_EMBED", BGE_M3)  # Default embedding model

# LLM models - listed smallest to largest (2GB-5GB-9GB in download size)
QWEN2_5_3B = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
QWEN2_5_7B = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/resolve/main/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
QWEN2_5_14B = "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-1M-GGUF/resolve/main/Qwen2.5-14B-Instruct-1M-Q4_K_M.gguf"
CURRENT_LLM = os.getenv("CURRENT_LLM", QWEN2_5_3B)  # Default LLM model

# LLM configuration
NUM_THREADS = int(os.getenv("NUM_THREADS", 4))  # Default thread count for LLM inference
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))  # Default temperature for responses

# System prompt for LLM
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
""")

# API Server configuration
HOST = os.getenv("HOST", "0.0.0.0")  # Listen on all interfaces
PORT = int(os.getenv("PORT", 8000))  # Default port
DEBUG = os.getenv("DEBUG", "False").lower() == "true"  # Debug mode
MAX_CONCURRENT_USERS = int(os.getenv("MAX_CONCURRENT_USERS", 30))  # Maximum number of concurrent WebSocket connections
QUEUE_TIMEOUT = int(os.getenv("QUEUE_TIMEOUT", 300))  # Seconds a user can stay in queue before timing out (5 minutes)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 120))  # Seconds to wait for a response before timing out (2 minutes)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "app.log"