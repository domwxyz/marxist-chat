import asyncio
import aiohttp
import json
import requests
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
from types import SimpleNamespace
import html
import logging
import re
import traceback

from queue import Queue, Empty
import threading
import time
import uuid

from llama_index.core import PromptTemplate, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

import config
from core.vector_store import VectorStoreManager
from core.llm_manager import LLMManager
from core.metadata_repository import MetadataRepository
from utils.redis_helper import RedisHelper

logger = logging.getLogger("core.query_engine")

class QueryProcessor:
    """Process queries in a dedicated thread to avoid concurrency issues"""
    def __init__(self, model):
        self.model = model
        self.request_queue = Queue()
        self.results = {}
        self.active = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("Query processor initialized with dedicated worker thread")
    
    def _process_queue(self):
        """Worker thread that processes queries sequentially"""
        while self.active:
            try:
                # Get the next request
                request_id, query_text, callback = self.request_queue.get(timeout=0.1)
                logger.info(f"Processing queued request {request_id}")
                
                # Process it
                try:
                    response = self.model.query(query_text)
                    # Store the response for streaming
                    self.results[request_id] = {
                        "response": response, 
                        "tokens": [], 
                        "complete": False,
                        "timestamp": time.time()
                    }
                    
                    # Stream tokens from the response
                    for token in response.response_gen:
                        self.results[request_id]["tokens"].append(token)
                        # Call optional progress callback
                        if callback:
                            callback(request_id, token)
                    
                    # Mark as complete
                    self.results[request_id]["complete"] = True
                    logger.info(f"Request {request_id} completed")
                    
                    # Store sources if available
                    if hasattr(response, 'source_nodes'):
                        self.results[request_id]["sources"] = response.source_nodes
                    elif hasattr(response, '_source_nodes'):
                        self.results[request_id]["sources"] = response._source_nodes
                        
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    self.results[request_id] = {"error": str(e), "complete": True}
                finally:
                    self.request_queue.task_done()
                    
                    # Clean up old results (keep for 30 minutes)
                    self._cleanup_old_results()
                    
            except Empty:
                pass  # No requests, continue checking
    
    def _cleanup_old_results(self):
        """Remove old results to prevent memory leaks"""
        now = time.time()
        expired_ids = []
        
        for req_id, result in self.results.items():
            # Keep results for 30 minutes
            if result.get("complete", False) and "timestamp" in result:
                if now - result["timestamp"] > 1800:  # 30 minutes
                    expired_ids.append(req_id)
        
        # Remove expired results
        for req_id in expired_ids:
            del self.results[req_id]
            
    def add_request(self, request_id, query_text, callback=None):
        """Add a query to the processing queue"""
        logger.info(f"Adding request {request_id} to queue")
        self.request_queue.put((request_id, query_text, callback))
        return request_id
        
    def get_tokens(self, request_id, last_index=0):
        """Get new tokens for a request"""
        if request_id not in self.results:
            return [], False, "Request not found"
            
        result = self.results[request_id]
        if "error" in result:
            return [], True, result["error"]
            
        new_tokens = result["tokens"][last_index:]
        return new_tokens, result["complete"], None
        
    def get_sources(self, request_id):
        """Get sources for a completed request"""
        if request_id not in self.results:
            return None
            
        result = self.results[request_id]
        if not result.get("complete", False):
            return None
            
        return result.get("sources", None)

class QueryEngine:
    def __init__(self, vector_store_dir=None):
        self.vector_store_manager = VectorStoreManager(vector_store_dir)
        self.query_engine = None
        self.streaming_engine = None
        self.index = None
        self.last_sources = []
        self.metadata_repository = MetadataRepository(self.vector_store_manager.cache_dir)
    
    def initialize(self) -> bool:
        """Initialize the query engine with the vector store and streaming only"""
        try:
            # Check if already initialized
            if self.query_engine is not None and hasattr(self, 'query_processor'):
                logger.info("Query engine already initialized")
                return True
                
            # Load vector store
            self.index = self.vector_store_manager.load_vector_store()
            if not self.index:
                logger.error("Failed to load vector store")
                return False
                
            # Load metadata repository
            success = self.metadata_repository.load_metadata_index()
            if not success:
                logger.warning("Failed to load metadata index - attempting to rebuild")
                success = self.metadata_repository.build_metadata_index()
                if not success:
                    logger.error("Failed to build metadata index")
                    # Continue anyway, as we can fall back to source node metadata
            
            # Create query template - using template that works with Qwen models
            query_template = PromptTemplate(
                "<|im_start|>system\n" + config.SYSTEM_PROMPT + "<|im_end|>\n"
                "<|im_start|>user\n"
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given this context, please answer the question: {query_str}\n"
                "Include guidance on which documents would be most helpful for the user to read for more information.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            # Create a streaming query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.35)
                ],
                text_qa_template=query_template,
                verbose=True,
                streaming=True
            )
            
            # Create a processor that will handle concurrent requests
            self.query_processor = QueryProcessor(self.query_engine)
            
            logger.info("Query engine and processor initialized successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def query(self, query_text: str, start_date=None, end_date=None):
        """Process a query with streaming and return the collected response"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call initialize() first.")
        
        # Reset sources for the new query
        self.last_sources = []
        
        # Get response
        try:
            logger.info(f"Processing query: {query_text[:100]}...")
            
            # Use streaming query engine
            streaming_response = self.query_engine.query(query_text)
            
            # Access the source nodes early if possible
            if hasattr(streaming_response, 'source_nodes'):
                self.last_sources = streaming_response.source_nodes
            elif hasattr(streaming_response, '_source_nodes'):
                self.last_sources = streaming_response._source_nodes
            
            # Collect all tokens from the streaming response to create a complete response
            response_text = ""
            for text in streaming_response.response_gen:
                response_text += text
                # Print the token for CLI feedback
                print(text, end="", flush=True)
                # We'll now return each token individually through our stream_query method
                # No need to collect them here
                
            # Check if response is empty or no sources were found
            if (not response_text.strip() or len(self.last_sources) == 0):
                print("\nDEBUG: Empty response from RAG, using metadata fallback")
                
                # Get metadata context
                metadata_context = self.metadata_repository.get_formatted_context()
                
                # Use direct LLM query with metadata context as fallback
                fallback_prompt = (
                    f"Based on the following document index, please provide information about: {query_text}\n\n"
                    f"{metadata_context}\n\n"
                    f"If you don't have relevant information in the document index, please state that clearly."
                )
                
                direct_response = Settings.llm.complete(fallback_prompt)
                response_text = direct_response.text if hasattr(direct_response, 'text') else str(direct_response)
            
            # Create a response-like object that has the .response attribute for compatibility
            response_obj = SimpleNamespace()
            response_obj.response = response_text
            response_obj.source_nodes = self.last_sources
            
            return response_obj
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            print(f"ERROR: {str(e)}")
            raise
    
    async def stream_query(self, query_text: str, stop_event: Optional[asyncio.Event] = None, start_date=None, end_date=None) -> AsyncGenerator[str, None]:
        """Process a query and stream response asynchronously with efficient concurrency"""

        if config.DISTRIBUTED_MODE:
            async for token in self.stream_query_distributed(query_text, stop_event, start_date, end_date):
                yield token
            return

        if not hasattr(self, 'query_processor') or not self.query_processor:
            raise ValueError("Query processor not initialized")
        
        # Reset sources for the new query
        self.last_sources = []
        
        try:
            # Generate a unique request ID
            request_id = f"req_{uuid.uuid4().hex[:8]}"
            
            # Add to queue processor
            self.query_processor.add_request(request_id, query_text)
            logger.info(f"Added request {request_id} to queue, now streaming responses")
            
            # Stream results as they become available
            last_index = 0
            while True:
                # Check for cancellation
                if stop_event and stop_event.is_set():
                    logger.info(f"Query {request_id} stopped by event")
                    return
                    
                # Get any new tokens
                new_tokens, complete, error = self.query_processor.get_tokens(request_id, last_index)
                
                # Yield any new tokens
                for token in new_tokens:
                    yield token
                    last_index += 1
                    
                    # Small sleep to prevent overwhelming the client
                    await asyncio.sleep(0.01)
                    
                    # Check for stop again after yielding
                    if stop_event and stop_event.is_set():
                        logger.info(f"Query {request_id} stopped after token")
                        return
                
                # If complete or error, we're done
                if complete:
                    # Store sources for future reference
                    sources = self.query_processor.get_sources(request_id)
                    if sources:
                        self.last_sources = sources
                    
                    if error:
                        logger.error(f"Error in query {request_id}: {error}")
                        yield f"\nError: {error}"
                    return
                    
                # No new tokens yet, wait a bit before checking again
                await asyncio.sleep(0.1)
                    
        except Exception as e:
            error_msg = f"Error in stream_query: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            yield f"\nError: {str(e)}"

    async def stream_query_distributed(self, query_text: str, stop_event: Optional[asyncio.Event] = None, start_date=None, end_date=None) -> AsyncGenerator[str, None]:
        """Process a query with the remote LLM service and stream the response asynchronously"""
        if stop_event is None:
            stop_event = asyncio.Event()
        
        # Create a unique request ID
        request_id = f"req_{uuid.uuid4().hex}"
        logger.info(f"Streaming query via LLM service: {request_id}")
        
        try:
            # Prepare the API request
            url = f"{config.LLM_SERVICE_URL}/query"
            payload = {
                "query_text": query_text,
                "request_id": request_id,
                "start_date": start_date,
                "end_date": end_date
            }
            
            # Create a persistent session for streaming
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=300) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"LLM service error: {response.status}, {error_text}")
                        yield f"Error from LLM service: {response.status}"
                        return
                    
                    # Process the streaming response
                    async for line in response.content:
                        # Check if the query was stopped
                        if stop_event.is_set():
                            # Send stop request to LLM service
                            try:
                                await session.post(
                                    f"{config.LLM_SERVICE_URL}/stop",
                                    json={"request_id": request_id}
                                )
                            except Exception as e:
                                logger.error(f"Error sending stop request: {e}")
                            return
                        
                        try:
                            line = line.decode('utf-8').strip()
                            if not line:
                                continue
                                
                            data = json.loads(line)
                            
                            # Extract and yield tokens
                            if "token" in data:
                                yield data["token"]
                            elif "sources" in data:
                                # Store sources for later retrieval
                                self.last_sources = data["sources"]
                            elif "error" in data:
                                logger.error(f"Error from LLM service: {data['error']}")
                                yield f"\nError: {data['error']}"
                                return
                            elif "status" in data and data["status"] == "stopped":
                                logger.info(f"Query {request_id} was stopped")
                                return
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from LLM service: {line}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing streaming response: {e}")
                            yield f"\nError processing response: {str(e)}"
                            return
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout in stream_query_distributed for request {request_id}")
            yield "\nError: Request timed out"
        except Exception as e:
            logger.error(f"Error in stream_query_distributed: {e}")
            logger.error(traceback.format_exc())
            yield f"\nError: {str(e)}"
    
    def format_response(self, response):
        """Format response with sources in a clean way"""
        output = [f"\nAnswer: {response.response}\n"]
        
        if response.source_nodes:
            output.append("\nSources:")
            seen_sources = set()  # Track unique sources
            source_counter = 1  # Explicit counter for sources
            
            for node in response.source_nodes:
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
                    output.append(f"\n{source_counter}. {title}")
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
                    
                    # Clean HTML entities in the excerpt
                    excerpt = html.unescape(excerpt)
                    
                    # Limit excerpt length and ensure it doesn't end with an incomplete word
                    if len(excerpt) > 200:
                        # Find the last space before 200 chars
                        last_space = excerpt[:200].rfind(' ')
                        if last_space > 150:  # Only truncate if we can get a reasonable chunk
                            excerpt = excerpt[:last_space] + "..."
                        else:
                            excerpt = excerpt[:200] + "..."
                    
                    output.append(f"   Relevant excerpt: {excerpt}\n")
                    
                    # Increment source counter
                    source_counter += 1
                    
                    # Limit to 5 unique sources
                    if source_counter > 5:
                        break
                        
                except Exception as e:
                    logger.error(f"Error formatting source: {str(e)}")
                    continue
            
        return "\n".join(output)
        
    def get_formatted_sources(self, max_sources=5):
        """Return formatted sources from the last query with robust metadata retrieval"""
        formatted_sources = []
        seen_urls = set()
        seen_filenames = set()
        
        # Ensure metadata repository is loaded
        if not self.metadata_repository.is_loaded:
            self.metadata_repository.load_metadata_index()
        
        # If we don't have source nodes, return empty list
        if not self.last_sources:
            logger.warning("No source nodes available")
            return formatted_sources
            
        for node in self.last_sources:
            try:
                # Extract filename using our helper method
                file_name = self._extract_filename_from_node(node)
                
                # Skip if we can't identify the file or already processed it
                if not file_name or file_name in seen_filenames:
                    continue
                seen_filenames.add(file_name)
                
                # Look up complete metadata from repository
                complete_metadata = self.metadata_repository.get_metadata_by_filename(file_name)
                
                if complete_metadata:
                    # Use complete metadata from repository
                    title = complete_metadata.get('title', 'Untitled')
                    date = complete_metadata.get('date', 'Unknown')
                    url = complete_metadata.get('url', 'No URL')
                    author = complete_metadata.get('author', 'Unknown')
                    feed_name = complete_metadata.get('feed_name', 'Unknown')
                    
                    # Skip duplicate URLs
                    if url in seen_urls and url != 'No URL':
                        continue
                    seen_urls.add(url)
                    
                    # Get excerpt from node text
                    excerpt = self._extract_excerpt_from_node(node)
                    
                    formatted_sources.append({
                        "title": title,
                        "date": date,
                        "author": author,
                        "url": url,
                        "feed_name": feed_name,
                        "excerpt": excerpt
                    })
                    
                    logger.debug(f"Added source with metadata from repository: {title}")
                    
                    if len(formatted_sources) >= max_sources:
                        break
                        
                    # Continue to next node
                    continue
                
                # If we get here, we couldn't find metadata in the repository
                # Try to extract it directly from the node text
                metadata = {}
                
                # First, check node.metadata for any useful information
                title = node.metadata.get('title', 'Untitled')
                date = node.metadata.get('date', 'Unknown')
                url = node.metadata.get('url', 'No URL')
                author = node.metadata.get('author', 'Unknown')
                feed_name = node.metadata.get('feed_name', 'Unknown')
                
                # Then try to extract from node text if available
                if hasattr(node, 'text') and node.text:
                    # Try to extract from embedded metadata block first
                    title_match = re.search(r'Title: (.*?)\n', node.text)
                    if title_match:
                        title = title_match.group(1).strip() or title
                    
                    date_match = re.search(r'Date: (.*?)\n', node.text)
                    if date_match:
                        date = date_match.group(1).strip() or date
                        
                    author_match = re.search(r'Author: (.*?)\n', node.text)
                    if author_match:
                        author = author_match.group(1).strip() or author
                        
                    url_match = re.search(r'URL: (.*?)\n', node.text)
                    if url_match:
                        url = url_match.group(1).strip() or url
                        
                    feed_match = re.search(r'Feed Source: (.*?)\n', node.text)
                    if feed_match:
                        feed_name = feed_match.group(1).strip() or feed_name
                
                # Skip documents with insufficient metadata
                if title == 'Untitled' and url == 'No URL':
                    logger.debug(f"Skipping node with insufficient metadata for file: {file_name}")
                    continue
                    
                # Skip duplicate URLs
                if url in seen_urls and url != 'No URL':
                    continue
                seen_urls.add(url)
                
                # Get excerpt from node text
                excerpt = self._extract_excerpt_from_node(node)
                
                formatted_sources.append({
                    "title": title,
                    "date": date, 
                    "author": author,
                    "url": url,
                    "feed_name": feed_name,
                    "excerpt": excerpt
                })
                
                logger.debug(f"Added source with extracted metadata: {title}")
                
                if len(formatted_sources) >= max_sources:
                    break
                    
            except Exception as e:
                logger.error(f"Error formatting source: {e}")
                logger.error(traceback.format_exc())
                continue
                
        # Final fallback - if we have no sources but have nodes, create basic entries
        if not formatted_sources and self.last_sources:
            logger.warning("Using fallback source formatting")
            for i, node in enumerate(self.last_sources[:max_sources]):
                try:
                    excerpt = self._extract_excerpt_from_node(node)
                    formatted_sources.append({
                        "title": f"Source {i+1}",
                        "date": "Unknown",
                        "author": "Unknown",
                        "url": "No URL",
                        "feed_name": "Unknown",
                        "excerpt": excerpt
                    })
                except Exception:
                    continue
        
        return formatted_sources
        
    def format_sources_only(self, source_nodes):
        """Format only the sources in a clean way"""
        output = ["\nSources:"]
        seen_sources = set()  # Track unique sources
        source_counter = 1  # Explicit counter for sources
        
        # Load the metadata repository if not already loaded
        if not self.metadata_repository.is_loaded:
            self.metadata_repository.load_metadata_index()

        for node in source_nodes:
            try:
                # First extract the filename to look up metadata
                file_name = self._extract_filename_from_node(node)
                
                # If we couldn't find a filename, try alternate extraction methods
                if not file_name:
                    # Try direct extraction from node text
                    node_metadata = self._extract_embedded_metadata(node.text)
                    content_lines = node.text.split('\n')
                    
                    # Look for file name in text
                    for line in content_lines:
                        if line.startswith("File:"):
                            file_name = line.split(":", 1)[1].strip()
                            break
                
                # Try to get complete metadata from repository
                metadata = {}
                if file_name:
                    complete_metadata = self.metadata_repository.get_metadata_by_filename(file_name)
                    if complete_metadata:
                        # Use metadata from repository
                        title = complete_metadata.get('title', 'Untitled')
                        date = complete_metadata.get('date', 'Unknown')
                        url = complete_metadata.get('url', 'No URL')
                        author = complete_metadata.get('author', 'Unknown')
                        feed_name = complete_metadata.get('feed_name', 'Unknown')
                        
                        source_id = f"{title}:{date}:{url}"
                        
                        # Skip duplicate sources
                        if source_id in seen_sources:
                            continue
                        seen_sources.add(source_id)
                        
                        # Format source information
                        output.append(f"\n{source_counter}. [{feed_name}] {title}")
                        output.append(f"   Date: {date}")
                        output.append(f"   Author: {author}")
                        output.append(f"   URL: {url}")
                        
                        # Extract and clean the content for the excerpt
                        excerpt = self._extract_excerpt_from_node(node)
                        output.append(f"   Relevant excerpt: {excerpt}\n")
                        
                        # Increment source counter
                        source_counter += 1
                        
                        # Limit to 5 unique sources
                        if source_counter > 5:
                            break
                        continue
                
                # Fallback to metadata extraction from node text if repository lookup failed
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
                feed_name = metadata.get('Feed Source') or node.metadata.get('feed_name', 'Unknown')
                
                source_id = f"{title}:{date}:{url}"
                
                # Skip duplicate sources
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                
                # Format source information
                output.append(f"\n{source_counter}. [{feed_name}] {title}")
                output.append(f"   Date: {date}")
                output.append(f"   Author: {author}")
                output.append(f"   URL: {url}")
                
                # Extract and clean the content for the excerpt
                excerpt = self._extract_excerpt_from_node(node)
                output.append(f"   Relevant excerpt: {excerpt}\n")
                
                # Increment source counter
                source_counter += 1
                
                # Limit to 5 unique sources
                if source_counter > 5:
                    break
                    
            except Exception as e:
                print(f"Error formatting source: {str(e)}")
                print(traceback.format_exc())
                continue
        
        return "\n".join(output)

    def _extract_embedded_metadata(self, text):
        """Extract metadata embedded in node text"""
        metadata = {}
        if not text:
            return metadata
            
        lines = text.split('\n')
        in_metadata = False
        
        for line in lines:
            if line.strip() == "---":
                if not in_metadata:
                    in_metadata = True
                    continue
                else:
                    break
                    
            if in_metadata and ": " in line:
                key, value = line.split(": ", 1)
                metadata[key.lower()] = value
        
        return metadata
        
    def _extract_filename_from_node(self, node):
        """Extract filename from node using multiple strategies"""
        # First try chunk_id
        chunk_id = node.metadata.get('chunk_id', '')
        if chunk_id and ':' in chunk_id:
            return chunk_id.split(':', 1)[0]
        
        # Try direct filename
        file_name = node.metadata.get('file_name', '')
        if file_name:
            return file_name
        
        # Try embedded metadata
        if hasattr(node, 'text') and node.text:
            # Try to extract from embedded metadata
            node_metadata = self._extract_embedded_metadata(node.text)
            file_name = node_metadata.get('file_name', '')
            if file_name:
                return file_name
            
            # Try to find it in the text
            file_match = re.search(r'File: (.*?)\n', node.text)
            if file_match:
                return file_match.group(1).strip()
            
            # Try to match by title if possible
            title_match = re.search(r'Title: (.*?)\n', node.text)
            if title_match and self.metadata_repository.is_loaded:
                title = title_match.group(1).strip()
                # Search metadata repository for this title
                for meta in self.metadata_repository.metadata_list:
                    if meta.get('title', '') == title:
                        return meta.get('file_name', '')
        
        # No filename found
        return ''
    
    def _extract_excerpt_from_node(self, node):
        """Helper method to extract a clean excerpt from a node"""
        excerpt = ""
        content_lines = node.text.split('\n')
        in_content = False
        
        for line in content_lines:
            if line.startswith('Content:'):
                in_content = True
                continue
            if in_content and line.strip():
                excerpt = line.strip()
                break
        
        if not excerpt:
            excerpt = node.text
        
        excerpt = excerpt.strip()
        if len(excerpt) > 200:
            excerpt = excerpt[:200] + "..."
            
        return excerpt
        
    def test_llm(self):
        """Test if the LLM is functioning properly"""
        try:
            test_prompt = "Generate a short response about communism in 1-2 sentences."
            print(f"DEBUG: Sending test prompt directly to LLM: {test_prompt}")
            response = Settings.llm.complete(test_prompt)
            print(f"DEBUG: Direct LLM response: {response}")
            return f"LLM Test Result: {response}"
        except Exception as e:
            print(f"ERROR: LLM test failed: {e}")
            print(traceback.format_exc())
            return f"LLM Test Error: {str(e)}"
        