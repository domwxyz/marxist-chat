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

2. You will be presented with a menu interface with the following options:
   - 1. Archive RSS Feed - Downloads articles from the RSS feed
   - 2. Create Vector Store - Creates the vector database from archived articles
   - 3. Load Vector Store - Loads the existing vector database
   - 4. Load Chat - Starts the chat interface
   - 5. Delete RSS Archive - Removes the downloaded articles
   - 6. Delete Vector Store - Removes the vector database
   - 7. Configuration - Adjust program settings
   - 0. Exit - Quit the program

3. First Time Setup:
   - Select option 1 to download the articles
   - Select option 2 to create the vector store
   - Select option 3 to load the vector store
   - Select option 4 to start chatting

4. Configuration Options (Option 7):
   - Change Embedding Model - Switch between BGE-M3 and GTE-Small
   - Change Chat Model - Choose between Qwen 2.5 models (3B, 7B, 14B)
   - Change Number of Threads - Adjust CPU thread usage (1-16)
   - Change Temperature - Adjust response creativity (0.0-1.0)
   - Add/Remove RSS Feeds - Manage RSS feed sources

5. Chat Interface:
   - Type your questions about the content when prompted
   - Type 'exit' to return to the main menu

## Models

The script supports multiple models:

### Chat Models (smallest to largest):
- Qwen 2.5 3B (Default) - ~2GB download
- Qwen 2.5 7B - ~5GB download
- Qwen 2.5 14B - ~9GB download

### Embedding Models:
- BGE-M3 (Default)
- GTE-Small

## Notes

- The script maintains a persistent cache of downloaded articles and vector store between runs
- You can use the menu options to manage the cache and vector store without manual file manipulation
- The configuration menu allows you to adjust model settings without editing the code
- Chat responses include relevant source information with titles, dates, authors, and URLs
- Press Ctrl+C to cancel operations or return to the menu

