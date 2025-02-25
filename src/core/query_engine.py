import asyncio
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
import logging
import traceback

from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor

import config
from core.vector_store import VectorStoreManager
from core.llm_manager import LLMManager

logger = logging.getLogger("core.query_engine")

class QueryEngine:
    def __init__(self, vector_store_dir=None):
        self.vector_store_manager = VectorStoreManager(vector_store_dir)
        self.query_engine = None
        self.streaming_engine = None
        self.index = None
        self.last_sources = []
    
    def initialize(self) -> bool:
        """Initialize the query engine with the vector store"""
        try:
            # Load vector store
            self.index = self.vector_store_manager.load_vector_store()
            if not self.index:
                logger.error("Failed to load vector store")
                return False
            
            # Create query template - using template that works with Qwen models
            query_template = PromptTemplate(
                "<|im_start|>system\n" + config.SYSTEM_PROMPT + "<|im_end|>\n"
                "<|im_start|>user\n"
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given this context, please answer the question: {query_str}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            # Create regular query engine (non-streaming for collecting sources)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.7)
                ],
                text_qa_template=query_template,
                streaming=False
            )
            
            # Create streaming query engine for streaming responses
            self.streaming_engine = self.index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.7)
                ],
                text_qa_template=query_template,
                streaming=True
            )
            
            logger.info("Query engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def query(self, query_text: str):
        """Process a query and return the response"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call initialize() first.")
        
        # Reset sources for the new query
        self.last_sources = []
        
        # Get response
        try:
            logger.info(f"Processing query: {query_text[:100]}...")
            response = self.query_engine.query(query_text)
            
            # Store sources for later retrieval
            if hasattr(response, 'source_nodes'):
                self.last_sources = response.source_nodes
                
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def stream_query(self, query_text: str, stop_event: Optional[asyncio.Event] = None) -> AsyncGenerator[str, None]:
        """Process a query and stream the response tokens asynchronously with stop capability"""
        if not self.streaming_engine:
            raise ValueError("Streaming engine not initialized. Call initialize() first.")
        
        # Reset sources for the new query
        self.last_sources = []
        
        try:
            # First, get sources synchronously to ensure we have them
            # We need to do this as a separate step since the streaming response may not 
            # provide source nodes correctly in all cases
            logger.info(f"Fetching sources for query: {query_text[:100]}...")
            non_streaming_response = self.query_engine.query(query_text)
            
            if hasattr(non_streaming_response, 'source_nodes'):
                self.last_sources = non_streaming_response.source_nodes
                logger.info(f"Found {len(self.last_sources)} source nodes for query")
            
            # Now, get streaming response
            logger.info(f"Processing streaming query: {query_text[:100]}...")
            response = self.streaming_engine.query(query_text)
            
            # Stream the response tokens
            buffer = ""
            chunk_size = 4  # Send in very small chunks for smooth streaming
            
            for text in response.response_gen:
                # Check for stop event if provided
                if stop_event and stop_event.is_set():
                    logger.info("Query streaming stopped by stop event")
                    return
                    
                buffer += text
                
                while len(buffer) >= chunk_size:
                    chunk = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]
                    yield chunk
                    
                    # Check for stop again after yielding
                    if stop_event and stop_event.is_set():
                        logger.info("Query streaming stopped by stop event after chunk")
                        return
                    
                    # Small sleep to prevent overwhelming the client
                    # This helps with smoother streaming experience
                    await asyncio.sleep(0.01)
            
            # Send any remaining text in the buffer
            if buffer:
                yield buffer
                
        except Exception as e:
            error_msg = f"Error processing streaming query: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            yield f"Error: {str(e)}"
    
    def format_response(self, response):
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
                    logger.error(f"Error formatting source {idx}: {str(e)}")
                    continue
        
        return "\n".join(output)
    
    def get_formatted_sources(self, max_sources=5):
        """Return formatted sources from the last query in a format suitable for API responses"""
        formatted_sources = []
        seen_urls = set()
        
        for node in self.last_sources:
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
                
                # Skip duplicate URLs
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Extract relevant text snippet
                excerpt = ""
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
                
                formatted_sources.append({
                    "title": title,
                    "date": date,
                    "author": author,
                    "url": url,
                    "excerpt": excerpt
                })
                
                # Limit to max_sources sources
                if len(formatted_sources) >= max_sources:
                    break
                    
            except Exception as e:
                logger.error(f"Error formatting source: {e}")
                continue
                
        return formatted_sources
        