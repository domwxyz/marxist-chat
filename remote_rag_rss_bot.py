import feedparser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, PromptTemplate, Document, load_index_from_storage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.faiss import FaissVectorStore
from pathlib import Path
import chardet
import unicodedata
import faiss
import json
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
    # Extract author information
    author = "Unknown"
    if hasattr(entry, 'author'):
        author = entry.author
    elif hasattr(entry, 'authors') and entry.authors:
        author = entry.authors[0].name if hasattr(entry.authors[0], 'name') else entry.authors[0]

    # Extract and parse date
    published_date = "No Date"
    if hasattr(entry, 'published'):
        try:
            # First try RFC 822 format
            parsed_time = time.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z")
            published_date = time.strftime("%Y-%m-%d %H:%M:%S", parsed_time)
        except ValueError:
            try:
                # Fallback to parsing published_parsed tuple
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = time.strftime("%Y-%m-%d %H:%M:%S", entry.published_parsed)
            except Exception as e:
                print(f"Date parsing error: {e}")
    
    # Extract categories/tags
    categories = []
    if hasattr(entry, 'tags'):
        categories = [tag.term for tag in entry.tags]
    elif hasattr(entry, 'categories'):
        categories = [cat for cat in entry.categories]
    
    # Get URL and title
    url = entry.link if hasattr(entry, 'link') else "No URL"
    title = entry.title if hasattr(entry, 'title') else "No Title"
    
    return {
        "title": title,
        "date": published_date,
        "author": author,
        "url": url,
        "categories": categories
    }


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
    
    # Remove "The post ... appeared first on ..." text
    content = re.sub(r'The post.*?appeared first on.*?\.', '', content)
    
    # Remove links to the main site
    content = re.sub(r'<a href="https://communistusa\.org[^>]*>.*?</a>', '', content)
    
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
    content = unicodedata.normalize('NFKD', content)
    
    # Clean up lines while preserving structure
    content = '\n'.join(line.strip() for line in content.splitlines())
    
    return content.strip()
    
def format_metadata(metadata):
    """Format metadata in a clean, structured way"""
    metadata_block = [
        "---",
        f"Title: {metadata['title']}",
        f"Date: {metadata['date']}",
        f"Author: {metadata['author']}",
        f"URL: {metadata['url']}"
    ]
    
    # Add categories if present
    if metadata['categories']:
        metadata_block.append(f"Categories: {', '.join(metadata['categories'])}")
    
    metadata_block.append("---\n")
    
    return "\n".join(metadata_block)

def cache_entries(entries):
    """Store entries in local cache as text files with proper UTF-8 encoding"""
    CACHE_DIR.mkdir(exist_ok=True)
    documents = []
    
    for entry in entries:
        try:
            # Extract metadata from the full feedparser entry
            metadata = extract_metadata_from_entry(entry)
            
            # Generate safe filename with UTF-8 support
            date_prefix = metadata['date'][:10]
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
            
            # Ensure all text is properly encoded
            metadata_formatted = format_metadata({
                k: ensure_unicode(v) for k, v in metadata.items()
            })
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
            
            # Create document object with explicit encoding
            doc = Document(
                text=document_text,
                metadata=metadata
            )
            documents.append(doc)
            
            # Write to file with explicit UTF-8 encoding
            with open(CACHE_DIR / filename, "w", encoding="utf-8", errors="replace") as f:
                f.write(document_text)
                
        except Exception as e:
            print(f"Error processing entry {entry.get('title', 'unknown')}: {str(e)}")
            continue

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
        return unicodedata.normalize('NFKD', text)
        
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
    
    # Setup embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=CURRENT_EMBED,
        max_length=256,
        embed_batch_size=32
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
            # Add error handling for file encoding
            for file in VECTOR_STORE_DIR.glob("*.json"):
                try:
                    # First try to read with UTF-8
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # If UTF-8 fails, try to detect encoding
                    with open(file, 'rb') as f:
                        raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected['encoding']
                    
                    # Rewrite file with UTF-8 encoding
                    with open(file, 'r', encoding=encoding) as f:
                        content = f.read()
                    with open(file, 'w', encoding='utf-8') as f:
                        f.write(content)
            
            # Now try to load the index with corrected encoding
            storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
            index = load_index_from_storage(
                storage_context=storage_context,
                service_context=Settings
            )
            print("Successfully loaded existing index")
            return index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.7)
                ],
                text_qa_template=query_template
            )
            
        except Exception as e:
            print(f"Error loading existing index: {str(e)}")
            print("Cleaning up corrupted vector store...")
            try:
                # Remove corrupted vector store
                shutil.rmtree(VECTOR_STORE_DIR)
            except Exception as cleanup_error:
                print(f"Error cleaning up vector store: {str(cleanup_error)}")
    
    print("Creating new vector store...")
    return create_new_vector_store(embed_model, query_template)

def create_new_vector_store(embed_model, query_template):
    try:
        VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)
        
        d = 1024 if CURRENT_EMBED == BGE_M3 else 384
        faiss_index = faiss.IndexHNSWFlat(d, 32)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        documents = []
        for file_path in CACHE_DIR.glob("*.txt"):
            try:
                # Add check for empty files
                if file_path.stat().st_size == 0:
                    print(f"Warning: Empty file found: {file_path}")
                    continue
                    
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    print(f"Warning: File contains only whitespace: {file_path}")
                    continue
                documents.append(Document(text=text))
            except UnicodeDecodeError:
                print(f"Warning: Encoding issue with file {file_path}")
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        text = f.read()
                    documents.append(Document(text=text))
                except Exception as doc_error:
                    print(f"Error loading document {file_path}: {str(doc_error)}")
                    continue
        
        if not documents:
            raise ValueError("No documents could be loaded from cache directory")
            
        # Add check for minimum document length
        valid_documents = [doc for doc in documents if len(doc.text.strip()) > 50]
        if not valid_documents:
            raise ValueError("No valid documents found (all documents too short)")
        
        index = VectorStoreIndex.from_documents(
            valid_documents,  # Use filtered documents
            storage_context=storage_context,
            show_progress=True,
            service_context=Settings
        )
        
        print(f"Saving vector store at {VECTOR_STORE_DIR}")
        index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
        
        return index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ],
            text_qa_template=query_template
        )
        
    except Exception as e:
        print(f"Error creating new vector store: {str(e)}")
        if VECTOR_STORE_DIR.exists():
            try:
                shutil.rmtree(VECTOR_STORE_DIR)
            except Exception:
                pass
        raise

def create_new_index(embed_model):
    """Create a new vector store index"""
    d = 1024 if CURRENT_EMBED == BGE_M3 else 384
    # Use HNSW for faster approximate search
    faiss_index = faiss.IndexHNSWFlat(d, 32)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = SimpleDirectoryReader(CACHE_DIR).load_data()
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
        service_context=Settings
    )
    
    print(f"Saving vector store at {VECTOR_STORE_DIR}")
    index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
    return index
    
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
