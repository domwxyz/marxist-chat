import feedparser
from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, PromptTemplate, Document, load_index_from_storage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from pathlib import Path
import chardet
import unicodedata
import ast
import os
import re
import signal
import shutil
import sys
import time

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
QWEN2_5_14B = "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-1M-GGUF/resolve/main/Qwen2.5-14B-Instruct-1M-Q4_K_M.gguf"

# Set current model to use for embedding
CURRENT_EMBED = BGE_M3

# Set current LLM to use for chat
CURRENT_LLM = QWEN2_5_3B

# Set number of threads to use for running chat
NUM_THREADS = 4

# Set current Temperature to use for chat
TEMPERATURE = 0.2

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
    
def display_menu():
    """Display the main menu options"""
    print("\nRSS RAG Bot Menu")
    print("----------------")
    print("1. Archive RSS Feed")
    print("2. Create Vector Store")
    print("3. Load Vector Store")
    print("4. Load Chat")
    print("5. Delete RSS Archive")
    print("6. Delete Vector Store")
    print("7. Configuration")  # Add configuration option
    print("0. Exit")
    print("\nNote: For first time setup, run options 1, 2, 3, and 4 in order.")
    
def display_config_menu():
    """Display configuration options"""
    print("\nConfiguration Menu")
    print("-----------------")
    print("1. Change Embedding Model")
    print("2. Change Chat Model")
    print("3. Change Number of Threads")
    print("4. Change Temperature")
    print("5. Add/Remove RSS Feeds")
    print("0. Back to Main Menu")

def manage_configuration():
    """Manage program configuration"""
    global CURRENT_EMBED, CURRENT_LLM, NUM_THREADS, TEMPERATURE, RSS_FEED_URLS
    
    while True:
        display_config_menu()
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            break
            
        elif choice == '1':
            print("\nAvailable embedding models:")
            print(f"1. BGE-M3 ({BGE_M3})")
            print(f"2. GTE-Small ({GTE_SMALL})")
            model_choice = input("Choose model (1-2): ").strip()
            if model_choice == '1':
                CURRENT_EMBED = BGE_M3
            elif model_choice == '2':
                CURRENT_EMBED = GTE_SMALL
            print(f"Embedding model set to: {CURRENT_EMBED}")
            
        elif choice == '2':
            print("\nAvailable chat models:")
            print("1. Qwen 2.5 3B (Smallest)")
            print("2. Qwen 2.5 7B (Medium)")
            print("3. Qwen 2.5 14B (Largest)")
            model_choice = input("Choose model (1-3): ").strip()
            if model_choice == '1':
                CURRENT_LLM = QWEN2_5_3B
            elif model_choice == '2':
                CURRENT_LLM = QWEN2_5_7B
            elif model_choice == '3':
                CURRENT_LLM = QWEN2_5_14B
            print(f"Chat model set to: {CURRENT_LLM}")
            
        elif choice == '3':
            threads = input("\nEnter number of threads (1-16): ").strip()
            try:
                threads = int(threads)
                if 1 <= threads <= 16:
                    NUM_THREADS = threads
                    print(f"Threads set to: {NUM_THREADS}")
                else:
                    print("Invalid number of threads. Must be between 1 and 16.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == '4':
            temp = input("\nEnter temperature (0.0-1.0): ").strip()
            try:
                temp = float(temp)
                if 0.0 <= temp <= 1.0:
                    TEMPERATURE = temp
                    print(f"Temperature set to: {TEMPERATURE}")
                else:
                    print("Invalid temperature. Must be between 0.0 and 1.0.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == '5':
            print("\nCurrent RSS feeds:")
            for i, feed in enumerate(RSS_FEED_URLS, 1):
                print(f"{i}. {feed}")
            print("\nOptions:")
            print("1. Add feed")
            print("2. Remove feed")
            print("3. Back")
            feed_choice = input("Choose option: ").strip()
            
            if feed_choice == '1':
                new_feed = input("Enter new RSS feed URL: ").strip()
                if new_feed and new_feed not in RSS_FEED_URLS:
                    RSS_FEED_URLS.append(new_feed)
                    print("Feed added successfully!")
            elif feed_choice == '2':
                if len(RSS_FEED_URLS) > 1:
                    del_idx = input("Enter feed number to remove: ").strip()
                    try:
                        del_idx = int(del_idx) - 1
                        if 0 <= del_idx < len(RSS_FEED_URLS):
                            removed = RSS_FEED_URLS.pop(del_idx)
                            print(f"Removed feed: {removed}")
                        else:
                            print("Invalid feed number.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                else:
                    print("Cannot remove the last RSS feed.")
                    
        else:
            print("\nInvalid choice. Please try again.")
    
def archive_rss_feed():
    """Archive RSS feed entries"""
    if CACHE_DIR.exists():
        response = input("\nRSS archive already exists. Do you want to update it? (y/n): ")
        if response.lower() != 'y':
            return
            
    print("\nFetching RSS entries...")
    all_entries = []
    for feed_url in RSS_FEED_URLS:
        print(f"Processing {feed_url}")
        entries = fetch_rss_entries(feed_url)
        all_entries.extend(entries)
    
    print(f"Found {len(all_entries)} total entries")
    cache_entries(all_entries)
    print("\nRSS feed archived successfully!")

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
            # Check if we already have this entry
            if not any(e.link == entry.link for e in entries):
                # Keep the full feedparser entry
                entries.append(entry)
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
    
def extract_metadata_from_entry(entry):
    """Extract and format metadata from a feedparser entry"""
    title = entry.get('title', 'Untitled')
    published_date = entry.get('published', entry.get('pubDate', 'Unknown Date'))
    author = entry.get('author', 'Unknown Author')
    url = entry.get('link', 'No URL')
    formatted_date = format_date(published_date)
    
    # Initialize categories list instead of set
    categories = []
    
    # Process tags field
    if hasattr(entry, 'tags'):
        for tag in entry.tags:
            # Handle dictionary-style tags
            if isinstance(tag, dict):
                tag_text = tag.get('term', '')
            else:
                tag_text = str(tag)

            # Clean tag and add if valid
            if ',' in tag_text:
                # Split and clean individual categories
                for raw_cat in tag_text.split(','):
                    cleaned = clean_category(raw_cat)
                    if cleaned and cleaned not in categories:
                        categories.append(cleaned)
            else:
                cleaned = clean_category(tag_text)
                if cleaned and cleaned not in categories:
                    categories.append(cleaned)
    
    # Process category field, handling both single and multiple categories
    if hasattr(entry, 'category'):
        # Handle string or list
        if isinstance(entry.category, str):
            cleaned = clean_category(entry.category)
            if cleaned and cleaned not in categories:
                categories.append(cleaned)
        elif isinstance(entry.category, list):
            for cat in entry.category:
                cleaned = clean_category(cat)
                if cleaned and cleaned not in categories:
                    categories.append(cleaned)
    
    return {
        "title": title,
        "date": formatted_date,
        "author": author,
        "url": url,
        "categories": categories  # Now a normal list of strings
    }


def clean_category(category):
    """Clean and normalize a category string"""
    if not category:
        return ""
    
    # Force to string and handle various wrapper characters
    if not isinstance(category, str):
        category = str(category)
        
    # Remove brackets, quotes, etc. from the entire string
    category = category.strip('[]\'\"')
    category = category.strip()
    
    if not category:
        return ""
    
    # Remove all special characters except letters, numbers, spaces, and hyphens
    cleaned = re.sub(r'[^\w\s-]', '', category)
    
    # Normalize spaces and hyphens
    cleaned = re.sub(r'[-\s]+', ' ', cleaned).strip()
    
    # Skip empty or single-character categories
    if len(cleaned) <= 1:
        return ""
    
    # Title case each word
    return cleaned.title()

def sanitize_filename(title):
    """Sanitize title for filesystem safety with Unicode support"""
    if not title:
        return "untitled"
        
    # Ensure title is Unicode
    title = ensure_unicode(title)
    
    # Remove HTML tags if any
    clean_title = re.sub(r'<[^>]+>', '', title)
    
    # Replace invalid characters with underscores
    # Allow basic Latin + Cyrillic + common Unicode characters
    sanitized = re.sub(r'[^\w\s\-\u0400-\u04FF\u00C0-\u00FF]', '_', clean_title)
    
    # Replace spaces with single hyphens
    sanitized = re.sub(r'\s+', '-', sanitized)
    
    # Collapse multiple hyphens/underscores
    sanitized = re.sub(r'[-_]{2,}', '-', sanitized)
    
    # Trim to reasonable length (leaving room for date prefix)
    sanitized = sanitized.strip('-_ ')[:75]
    
    # Ensure we have something
    if not sanitized:
        return "untitled"
        
    return sanitized
    
def clean_rss_boilerplate(content):
    """Remove RSS feed boilerplate text and links"""
    if not content:
        return ""
    
    # Handle common RSS boilerplate patterns
    patterns = [
        # Patterns
        r'The post.*?appeared first on.*?\.',
        r'<a href="https://communistusa\.org[^>]*>.*?</a>',
        r'org">Revolutionary Communists of America\.',
        r'org">[^<]*</a>',  # Catches any remaining org"> patterns
        r'\[ ?to read the full analysis[^\]]*\]',  # Catches "[to read...]" patterns
        r'\[ ?read more[^\]]*\]',  # Catches "[read more...]" patterns
        r'Read more\.\.\..*$',  # Catches "Read more..." at end of content
        r'Continue reading.*$'  # Catches "Continue reading..." at end of content
    ]
    
    # Apply each cleanup pattern
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up any resulting blank lines from removed content
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Remove extra whitespace while preserving paragraph structure
    lines = [line.strip() for line in content.splitlines()]
    content = '\n'.join(line for line in lines if line)
    
    return content.strip()
    
def extract_content_sections(entry):
    """Extract and clean the description and main content with proper encoding"""
    # Get description from summary field
    description = entry.get('summary', '')
    if description:
        description = ensure_unicode(description)
        description = clean_rss_boilerplate(description)
        description = preprocess_content(description)
    
    # Get main content from content field
    content = ""
    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].get('value', '')
        content = ensure_unicode(content)
        content = clean_rss_boilerplate(content)
        content = preprocess_content(content)
    
    return description, content
    
def preprocess_content(content):
    """Clean and normalize content while preserving Unicode"""
    if not content:
        return ""
    
    # Ensure content is Unicode
    content = ensure_unicode(content)
    
    # Remove script and style elements
    content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
    content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
    
    # Handle WordPress image captions
    content = re.sub(r'<div[^>]*class="wp-caption[^>]*>.*?</div>', '', content, flags=re.DOTALL)
    
    # Convert HTML line breaks to newlines
    content = content.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    content = content.replace('</p>', '\n\n').replace('<p>', '')
    
    # Remove remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Fix common HTML entities with Unicode equivalents
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&mdash;': '—',
        '&ndash;': '–',
        '&hellip;': '…',
        '&#8217;': ''',
        '&#8216;': ''',
        '&#8220;': '"',
        '&#8221;': '"',
    }
    for entity, char in html_entities.items():
        content = content.replace(entity, char)
    
    # Normalize whitespace while preserving paragraphs
    content = re.sub(r'\s+([.,!?])', r'\1', content)
    content = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Normalize Unicode forms
    content = unicodedata.normalize('NFKC', content)  # Changed from NFKD to NFKC for better encoding stability
    
    # Clean up lines while preserving structure
    content = '\n'.join(line.strip() for line in content.splitlines())
    
    return content.strip()
    
def format_metadata(metadata):
    metadata_block = [
        "---",
        f"Title: {metadata['title']}",
        f"Date: {metadata['date']}",
        f"Author: {metadata['author']}",
        f"URL: {metadata['url']}"
    ]
    
    if metadata.get('categories'):
        categories_str = ', '.join(metadata['categories'])
        metadata_block.append(f"Categories: {categories_str}")
    
    metadata_block.append("---\n")
    return '\n'.join(metadata_block)
    
def format_date(date_str):
    """Format RSS date string to YYYY-MM-DD"""
    if not date_str or date_str == 'Unknown Date':
        return datetime.now().strftime('%Y-%m-%d')
        
    try:
        # RSS feeds typically use this format
        dt = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return datetime.now().strftime('%Y-%m-%d')

def cache_entries(entries):
    """Store entries in local cache as text files with proper UTF-8 encoding"""
    CACHE_DIR.mkdir(exist_ok=True)
    documents = []
    last_known_date = datetime.now().strftime('%Y-%m-%d')  # Default fallback
    
    for entry in entries:
        try:
            # Extract metadata from the full feedparser entry
            metadata = extract_metadata_from_entry(entry)
            
            # If we got a valid date, update our last_known_date
            if metadata['date'] and metadata['date'] != 'Unknown Date':
                last_known_date = metadata['date']
            else:
                # Use the last known date if this entry's date is missing/unknown
                metadata['date'] = last_known_date
                print(f"Using fallback date {last_known_date} for entry: {metadata['title']}")
            
            # Generate safe filename with UTC-8 support
            date_prefix = metadata['date']  # Will be either actual date or last_known_date
            safe_title = sanitize_filename(metadata['title'])
            filename = f"{date_prefix}_{safe_title}.txt"
            
            # Ensure uniqueness
            counter = 1
            while (CACHE_DIR / filename).exists():
                filename = f"{date_prefix}_{safe_title}-{counter}.txt"
                counter += 1
            
            # Add filename to metadata
            metadata['file_name'] = filename
            
            # Extract content sections with explicit encoding handling
            description, content = extract_content_sections(entry)
            
            # Create metadata text block, ensuring categories are formatted properly
            # This is the critical fix: directly use the format_metadata function with raw metadata
            # instead of ensuring_unicode on each value, which corrupts the list structure
            metadata_formatted = format_metadata(metadata)
            
            # Ensure text content is properly encoded
            description = ensure_unicode(description)
            content = ensure_unicode(content)
            
            # Combine content with clear section markers
            full_content = [
                metadata_formatted,
                "Description:",
                description,
                "\nContent:",
                content or description  # Use description as content if no content available
            ]
            
            # Join with proper line endings
            document_text = '\n'.join(full_content)
            
            # Create document object
            doc = Document(
                text=document_text,
                metadata=metadata  # Use the original metadata object
            )
            documents.append(doc)
            
            # Write to file with explicit UTF-8 encoding
            with open(CACHE_DIR / filename, "w", encoding="utf-8", errors="replace") as f:
                f.write(document_text)
                
        except Exception as e:
            print(f"Error processing entry {entry.get('title', 'unknown')}: {str(e)}")
            continue

    return documents

def ensure_unicode(text):
    if text is None:
        return ""
        
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Add fallback for empty or invalid raw_data
                raw_data = text if text else b''
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] if detected and detected['encoding'] else 'utf-8'
                return text.decode(encoding, errors='replace')
            except Exception:
                return text.decode('utf-8', errors='replace')
    
    if isinstance(text, str):
        return unicodedata.normalize('NFKC', text)
        
    return str(text)

def write_documents_batch(documents):
    """Write a batch of documents to files"""
    for doc in documents:
        try:
            with open(CACHE_DIR / doc.metadata["file_name"], "w", encoding="utf-8") as f:
                f.write(doc.text)
        except Exception as e:
            print(f"Error writing document {doc.metadata.get('file_name', 'unknown')}: {str(e)}")
            continue

def initialize_llm():
    """Initialize the LLAMA CPP LLM with model-specific settings"""
    model_url = CURRENT_LLM
    
    # Default configuration for Qwen models
    return LlamaCPP(
        model_url=model_url,
        model_path=None,
        temperature=TEMPERATURE,
        max_new_tokens=1024,
        context_window=8192,
        model_kwargs={"n_threads": NUM_THREADS},
        verbose=False
    )

def setup_rag_system():
    """Initialize the RAG pipeline with vector store and robust encoding handling"""
    llm = initialize_llm()
    print("\nChat model initialized.")
    
    # Setup embedding model with optimized parameters
    embed_model = HuggingFaceEmbedding(
        model_name=CURRENT_EMBED,
        max_length=512,
        embed_batch_size=64
    )
    print("Embedding model initialized.")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=25,
        paragraph_separator="\n\n",
        tokenizer=None
    )
    
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
            storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
            index = load_index_from_storage(
                storage_context=storage_context,
                service_context=Settings
            )
            print("Successfully loaded existing index")
        except Exception as e:
            print(f"Error loading existing index: {str(e)}")
            print("Creating new vector store...")
            index = create_vector_store()
            if not index:
                raise ValueError("Failed to create vector store")
    else:
        print("Creating new vector store...")
        index = create_vector_store()
        if not index:
            raise ValueError("Failed to create vector store")
            
    return index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
        text_qa_template=query_template
    )

def create_vector_store():
    """Create a new vector store with simpler in-memory approach"""
    if not CACHE_DIR.exists() or not any(CACHE_DIR.iterdir()):
        print("\nError: No RSS archive found. Please run option 1 first.")
        return None
        
    if VECTOR_STORE_DIR.exists():
        response = input("\nVector store already exists. Do you want to recreate it? (y/n): ")
        if response.lower() != 'y':
            return None
        try:
            shutil.rmtree(VECTOR_STORE_DIR)
        except Exception as e:
            print(f"\nError deleting existing vector store: {e}")
            return None
            
    print("\nInitializing embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name=CURRENT_EMBED,
        max_length=512,
        embed_batch_size=64
    )
    
    # Set the embedding model in Settings before creating the index
    Settings.embed_model = embed_model
    print("Embedding model initialized and set.")
    
    try:
        VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)
        
        # Use default storage context (SimpleVectorStore)
        storage_context = StorageContext.from_defaults()
        
        # Process all documents first
        print("Loading documents...")
        all_documents = []
        total_files = len(list(CACHE_DIR.glob("*.txt")))
        
        for i, file_path in enumerate(CACHE_DIR.glob("*.txt"), 1):
            try:
                if file_path.stat().st_size == 0:
                    continue
                    
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                    
                if not text.strip():
                    continue
                    
                # Apply consistent normalization for all text content
                text = unicodedata.normalize('NFKC', text)
                all_documents.append(Document(text=text))
                
                if i % 100 == 0:  # Progress update every 100 files
                    print(f"Loaded {i}/{total_files} documents...")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Filter valid documents
        valid_documents = [doc for doc in all_documents if len(doc.text.strip()) > 50]
        print(f"\nProcessing {len(valid_documents)} valid documents...")
        
        # Set service context with our embedding model
        service_context = Settings
        
        # Create index from all valid documents at once
        index = VectorStoreIndex.from_documents(
            valid_documents,
            storage_context=storage_context,
            show_progress=True,
            service_context=service_context
        )
        
        print(f"\nSaving vector store at {VECTOR_STORE_DIR}")
        index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
        print("\nVector store created successfully!")
        
        return index
        
    except Exception as e:
        print(f"\nError creating vector store: {e}")
        if VECTOR_STORE_DIR.exists():
            try:
                shutil.rmtree(VECTOR_STORE_DIR)
            except Exception:
                pass
        return None
        
def load_vector_store():
    """Load existing vector store"""
    if not VECTOR_STORE_DIR.exists():
        print("\nError: No vector store found. Please run option 2 first.")
        return None
        
    print("\nLoading vector store...")
    try:
        query_engine = setup_rag_system()
        print("Vector store loaded successfully!")
        return query_engine
    except Exception as e:
        print(f"\nError loading vector store: {e}")
        return None
    
def load_chat(query_engine):
    """Start the chat interface"""
    if not query_engine:
        print("\nError: No query engine loaded. Please run option 3 first.")
        return
        
    print("\nStarting chat interface. Type 'exit' to return to menu.")
    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            continue
        if query.lower() == 'exit':
            break
            
        try:
            response = query_engine.query(query)
            formatted_output = format_response(response)
            print(formatted_output)
        except Exception as e:
            print(f"\nError: {e}")
            
def delete_rss_archive():
    """Delete the RSS archive directory"""
    if not CACHE_DIR.exists():
        print("\nNo RSS archive found.")
        return
        
    response = input("\nAre you sure you want to delete the RSS archive? (y/n): ")
    if response.lower() == 'y':
        try:
            shutil.rmtree(CACHE_DIR)
            print("\nRSS archive deleted successfully!")
        except Exception as e:
            print(f"\nError deleting RSS archive: {e}")

def delete_vector_store():
    """Delete the vector store directory"""
    if not VECTOR_STORE_DIR.exists():
        print("\nNo vector store found.")
        return
        
    response = input("\nAre you sure you want to delete the vector store? (y/n): ")
    if response.lower() == 'y':
        try:
            shutil.rmtree(VECTOR_STORE_DIR)
            print("\nVector store deleted successfully!")
        except Exception as e:
            print(f"\nError deleting vector store: {e}")
    
def format_response(response):
    """Format response with sources in a clean way"""
    output = [f"\nAnswer: {response.response}\n"]
    
    if response.source_nodes:
        output.append("\nSources:")
        seen_sources = set()  # Track unique sources
        
        for idx, node in enumerate(response.source_nodes, 1):
            try:
                # Extract metadata from the actual content
                content_lines = node.text.split('\n')
                metadata = {}
                
                # Parse metadata section
                in_metadata = False
                for line in content_lines:
                    if line.strip() == '---':
                        if not in_metadata:
                            in_metadata = True
                            continue
                        else:
                            break
                    if in_metadata:
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            metadata[key] = value
                
                # Use extracted metadata or fallback to node.metadata
                title = metadata.get('Title') or node.metadata.get('title', 'Untitled')
                date = metadata.get('Date') or node.metadata.get('date', 'Unknown')
                url = metadata.get('URL') or node.metadata.get('url', 'No URL')
                author = metadata.get('Author') or node.metadata.get('author', 'Unknown')
                
                source_id = f"{title}:{date}:{url}"
                
                # Skip duplicate sources
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                
                # Format source information
                output.append(f"\n{idx}. {title}")
                output.append(f"   Date: {date}")
                output.append(f"   Author: {author}")
                output.append(f"   URL: {url}")
                
                categories_str = metadata.get('Categories') or ''
                categories_list = [cat.strip() for cat in categories_str.split(',') if cat.strip()]
            
                if categories_list:
                    output.append(f"   Categories: {', '.join(categories_list)}")
                
                # Extract and clean the content for the excerpt
                content_section = ""
                in_content = False
                for line in content_lines:
                    if line.startswith('Content:'):
                        in_content = True
                        continue
                    if in_content and line.strip():
                        content_section = line.strip()
                        break
                
                # Use content section for excerpt, fallback to raw text if needed
                excerpt = content_section if content_section else node.text
                excerpt = excerpt.strip()
                if len(excerpt) > 200:
                    excerpt = excerpt[:200] + "..."
                
                output.append(f"   Relevant excerpt: {excerpt}\n")
                
                # Limit to 5 unique sources
                if len(seen_sources) >= 5:
                    break
                    
            except Exception as e:
                print(f"Error formatting source {idx}: {str(e)}")
                continue
    
    return "\n".join(output)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    query_engine = None
    
    while True:
        try:
            display_menu()
            choice = input("\nEnter your choice (0-7): ").strip()  # Update range
            
            if choice == '0':
                print("\nGoodbye!")
                break
                
            elif choice == '1':
                archive_rss_feed()
                
            elif choice == '2':
                create_vector_store()
                
            elif choice == '3':
                query_engine = load_vector_store()
                
            elif choice == '4':
                load_chat(query_engine)
                
            elif choice == '5':
                delete_rss_archive()
                
            elif choice == '6':
                delete_vector_store()
                query_engine = None  # Reset query engine if vector store is deleted
                
            elif choice == '7':  # Add configuration option
                manage_configuration()
                
            else:
                print("\nInvalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
