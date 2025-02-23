# marxist-chat
Chatbot for RCA articles at communistusa.org

A Python script that creates a local RAG (Retrieval Augmented Generation) chatbot from RSS feed content using a mix of Llama Index and LlamaCPP. Currently configured to download and interact with RCA's articles from the communistusa.org RSS feed.

## Requirements

- Python 3.8+
- Required Python packages:
  ```
  llama-index-core
  llama-index-llms-llama-cpp
  llama-index-embeddings-huggingface
  llama-index-vector-stores-faiss
  faiss-cpu
  feedparser
  chardet
  ```

## Installation


Install the required packages:
```bash
pip install llama-index-core llama-index-llms-llama-cpp llama-index-embeddings-huggingface llama-index-vector-stores-faiss faiss-cpu feedparser chardet
```

## Usage

1. Run the script:
   ```bash
   python remote_rag_rss_bot.py
   ```

2. The script will:
   - Download all articles from the RCA RSS feed into a `posts_cache` folder
   - Create a vector store database in the `vector_store` folder
   - Start an interactive chat interface

3. Chat with the bot by typing questions about the content when prompted

## Rebuilding the Database

To re-archive the articles and rebuild the database:

1. Delete both the `posts_cache` and `vector_store` folders
2. Run the script again - it will recreate both folders with fresh content

## Notes

- The script uses Qwen 3B as the default chat model and BGE-M3 for embeddings
   - This can be changed by switching the CURRENT_LLM and CURRENT_EMBED constants at the top of the script
- Cached content and vector store are persistent between runs unless manually deleted
- Press Ctrl+C to exit the chat interface

