import feedparser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, PromptTemplate, load_index_from_storage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from pathlib import Path
import os
import re
import time
import signal
import shutil
import sys

# Configuration
RSS_FEED_URLS = [
    "https://communistusa.org/feed",
    # Add more feeds here
]
CACHE_DIR = Path("./posts_cache")
VECTOR_STORE_DIR = Path("./vector_store")

# Embedding models listed smallest to largest
BGE_M3 = "BAAI/bge-m3"
GTE_SMALL = "thenlper/gte-small"

# Chat models listed smallest to largest (2GB-5GB-9GB in download size)
QWEN2_5_3B = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
QWEN2_5_7B = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/resolve/main/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
PHI_4 = "https://huggingface.co/bartowski/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf"

# Set current model to use for embedding
CURRENT_EMBED = BGE_M3

# Set current LLM to use for chat
CURRENT_LLM = QWEN2_5_3B

# Set system prompt for chatbot instance
SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

def signal_handler(sig, frame):
    """Handle CTRL + C exit signal"""
    print("\nGoodbye!")
    sys.exit(0)

def fetch_rss_entries(feed_url):
    """Fetch all entries from WordPress RSS feed with pagination"""
    entries = []
    page = 1
    has_more = True
    
    while has_more:
        # Handle WordPress pagination pattern
        current_url = f"{feed_url.rstrip('/')}/?paged={page}" if page > 1 else feed_url
        feed = feedparser.parse(current_url)
        print(f"Fetching page {page}...")
        
        if not feed.entries:
            has_more = False
            break
            
        new_entries = 0
        for entry in feed.entries:
            content = "\n".join([c.value for c in entry.content]) if entry.content else ""
            if not any(e.get("link") == entry.link for e in entries):
                entries.append({
                    "title": entry.title,
                    "description": entry.description,
                    "link": entry.link,
                    "published": time.strftime("%Y-%m-%d %H:%M:%S", entry.published_parsed),
                    "content": content
                })
                new_entries += 1
                
        # WordPress pagination fallback logic
        if new_entries == 0:
            has_more = False
        else:
            page += 1

        # Check for standard RSS pagination links as backup
        next_page = None
        for link in feed.feed.get("links", []):
            if link.rel == "next":
                next_page = link.href
                break
        
        if next_page and next_page != current_url:
            feed_url = next_page  # Update base URL if different
            page = 1  # Reset page counter
            
        time.sleep(0.2)  # Respect server resources
        
    print("Finished fetching all pages.")
    return entries

def sanitize_filename(title):
    """Sanitize title for filesystem safety"""
    # Remove HTML tags if any
    clean_title = re.sub(r'<[^>]+>', '', title)
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\u0400-\u04FF\-_\. ]', '_', clean_title)  # Allows basic Latin + Cyrillic
    
    # Replace spaces with single hyphens
    sanitized = re.sub(r'\s+', '-', sanitized)
    
    # Collapse multiple hyphens/underscores
    sanitized = re.sub(r'[-_]{2,}', '-', sanitized)
    
    # Trim to 75 characters (leaving room for date prefix)
    return sanitized.strip('-_ ')[:75]

def cache_entries(entries):
    """Store entries in local cache as text files with sanitized names"""
    CACHE_DIR.mkdir(exist_ok=True)
    
    for entry in entries:
        # Generate safe filename
        date_prefix = entry['published'][:10]
        safe_title = sanitize_filename(entry['title'])
        filename = f"{date_prefix}_{safe_title}.txt"
        
        # Add metadata structure
        metadata = {
            "title": entry['title'],
            "date": entry['published'],
            "url": entry['link'],
            "source": "wordpress_rss"
        }
        
        # Write content with better structure
        content = (
            f"{entry['title']}\n\n"
            f"{entry['description']}\n\n"
            f"{entry['content']}"
        )
        
        # Create document with metadata
        doc = Document(
            text=content,
            metadata=metadata
        )

def initialize_llm():
    """Initialize the LLAMA CPP LLM with model-specific settings"""
    model_url = CURRENT_LLM
    
    # Model-specific configurations
    if "phi-4" in model_url.lower():
        return LlamaCPP(
            model_url=model_url,
            model_path=None,
            temperature=0.2,
            max_new_tokens=1024,
            context_window=8192,
            model_kwargs={"n_threads": 4},
            verbose=False,
            # Add specific prompt template for Phi-4
            messages_to_prompt=lambda messages: (
                "Instruct: Based on the given context, " + messages[-1].content + "\nOutput: "
            ),
            completion_to_prompt=lambda completion: completion + "\n"
        )
    else:
        # Default configuration for Qwen models
        return LlamaCPP(
            model_url=model_url,
            model_path=None,
            temperature=0.2,
            max_new_tokens=1024,
            context_window=8192,
            model_kwargs={"n_threads": 4},
            verbose=False
        )

def setup_rag_system():
    """Initialize the RAG pipeline with vector store"""
    llm = initialize_llm()
    print("\nChat model intialized.")
    
    # Setup embedding model
    embed_model = HuggingFaceEmbedding(model_name=CURRENT_EMBED)
    print("Embedding model initialized.")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separator=" ",
        paragraph_separator="\n\n",
        secondary_delimiter=[". ", "? ", "! "],
    )
    
    # Model-specific query template
    if "phi-4" in CURRENT_LLM.lower():
        query_template = PromptTemplate(
            "System: " + SYSTEM_PROMPT + "\n\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given this context, please answer the question: {query_str}\n"
        )
    else:
        # Template for Qwen models
        query_template = PromptTemplate(
            "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given this context, please answer the question: {query_str}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    # Check for existing vector store
    if VECTOR_STORE_DIR.exists():
        print(f"Loading existing vector store from {VECTOR_STORE_DIR}")
        try:
            # Load the full index with all components
            index = load_index_from_storage(
                storage_context=StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR),
                llm=llm,
                embed_model=embed_model
            )
            print("Successfully loaded existing index")
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Creating new vector store...")
            # If loading fails, create new index
            storage_context = StorageContext.from_defaults()
            documents = SimpleDirectoryReader(CACHE_DIR).load_data()
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
                llm=llm,
                embed_model=embed_model
            )
            # Save the new index
            print(f"Saving vector store at {VECTOR_STORE_DIR}")
            index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
    else:
        print("Creating new vector store...")
        storage_context = StorageContext.from_defaults()
        documents = SimpleDirectoryReader(CACHE_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
            llm=llm,
            embed_model=embed_model
        )
        print(f"Saving vector store at {VECTOR_STORE_DIR}")
        index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
    
    print("Initializing query engine.")
    return index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
        text_qa_template=query_template
    )
    
def format_response(response):
    """Format response with sources in a clean way"""
    output = [f"\nAnswer: {response.response}\n"]
    
    if response.source_nodes:
        output.append("\nSources:")
        for idx, node in enumerate(response.source_nodes[:5], 1):
            metadata = node.metadata
            output.append(f"\n{idx}. {metadata.get('title', 'Untitled')}")
            output.append(f"   Date: {metadata.get('date', 'Unknown')}")
            output.append(f"   URL: {metadata.get('url', 'No URL')}")
            output.append(f"   Relevant excerpt: {node.text[:200]}...\n")
    
    return "\n".join(output)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    # Step 1: Fetch and cache RSS content
    if not CACHE_DIR.exists():
        print("Fetching RSS entries...")
        all_entries = []
        for feed_url in RSS_FEED_URLS:
            print(f"Processing {feed_url}")
            entries = fetch_rss_entries(feed_url)
            all_entries.extend(entries)
        
        print(f"Found {len(all_entries)} total entries")
        cache_entries(all_entries)
    else:
        print("\nExisting post cache found. Skipping RSS fetch.")
    
    # Step 2: Initialize RAG system
    print("\nInitializing RAG system...")
    query_engine = setup_rag_system()
    
    # Step 3: Chat interface
    print("\nRAG system ready! Ask questions about the content. Press Ctrl+C to exit.")
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            if not query:
                continue
                
            # Get response with sources
            response = query_engine.query(query)
            
            # Format and print the response
            formatted_output = format_response(response)
            print(formatted_output)
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
