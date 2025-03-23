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

# RSS feed configuration with pagination types
RSS_FEED_CONFIG = [
    {"url": "https://communistusa.org/feed", "pagination_type": "wordpress"},
    {"url": "https://marxist.com/index.php?format=feed", "pagination_type": "joomla", "limit_increment": 5},
    {"url": "https://communist.red/feed", "pagination_type": "wordpress"}, 
    # Add more feeds here with appropriate pagination_type and settings
]

# For backwards compatibility
DEFAULT_RSS_FEEDS = [feed["url"] for feed in RSS_FEED_CONFIG]
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
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a knowledgeable assistant specializing in Marxist and communist theory and practice. Your primary role is to provide concise, accurate answers based on the source documents and guide users to relevant articles for deeper reading.

ANSWER GUIDELINES:
1. Give brief, accurate answers (2-3 sentences) that directly address the question
2. Always include which documents contain more detailed information on the topic
3. When multiple documents offer different perspectives, note these distinctions
4. If a topic has evolved over time, mention how the understanding has changed
5. When recommending documents, include their titles and dates

NEVER:
- Generate lengthy explanations or summaries
- Make up information not contained in the documents
- Present yourself as having personal opinions about communism
- Introduce your responses with phrases like "Based on the documents" or "According to the sources"

ALWAYS:
- Be helpful, direct, and factual
- Direct users to specific documents for more information
- Maintain a neutral, informative tone
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
