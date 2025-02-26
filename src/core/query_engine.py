import asyncio
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
import logging
import traceback

from llama_index.core import PromptTemplate, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

import config
from core.vector_store import VectorStoreManager
from core.llm_manager import LLMManager
from core.metadata_repository import MetadataRepository

logger = logging.getLogger("core.query_engine")

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
            # Load vector store
            self.index = self.vector_store_manager.load_vector_store()
            if not self.index:
                logger.error("Failed to load vector store")
                return False
                
            self.metadata_repository.load_metadata_index()
            
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
            
            # Create a streaming query engine only
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=4,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.4)
                ],
                text_qa_template=query_template,
                verbose=True,
                streaming=True  # Always streaming
            )
            
            logger.info("Streaming query engine initialized successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def query(self, query_text: str):
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
            
            # Collect all tokens from the streaming response to create a complete response
            response_text = ""
            for text in streaming_response.response_gen:
                response_text += text
                # Print the token for CLI feedback
                print(text, end="", flush=True)
            
            # Store sources for later retrieval
            if hasattr(streaming_response, 'source_nodes'):
                self.last_sources = streaming_response.source_nodes
            elif hasattr(streaming_response, '_source_nodes'):
                # Some versions of llama-index store source nodes in a private attribute
                self.last_sources = streaming_response._source_nodes
            else:
                logger.warning("No source nodes found in streaming response")
            
            # Check if response is empty or no sources were found
            if (not response_text.strip() or len(self.last_sources) == 0):
                print("\nDEBUG: Empty response from RAG, using metadata fallback")
                
                # Get metadata context
                metadata_context = self.metadata_repository.get_formatted_context()
                
                # Use direct LLM query with metadata context as fallback
                from llama_index.core import Settings
                fallback_prompt = (
                    f"Based on the following document index, please provide information about: {query_text}\n\n"
                    f"{metadata_context}\n\n"
                    f"If you don't have relevant information in the document index, please state that clearly."
                )
                
                direct_response = Settings.llm.complete(fallback_prompt)
                response_text = direct_response.text if hasattr(direct_response, 'text') else str(direct_response)
            
            # Create a response-like object that has the .response attribute for compatibility
            from types import SimpleNamespace
            response_obj = SimpleNamespace()
            response_obj.response = response_text
            response_obj.source_nodes = self.last_sources
            
            return response_obj
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            print(f"ERROR: {str(e)}")
            raise
    
    async def stream_query(self, query_text: str, stop_event: Optional[asyncio.Event] = None) -> AsyncGenerator[str, None]:
        """Process a query and stream the response tokens asynchronously with stop capability"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call initialize() first.")
        
        # Reset sources for the new query
        self.last_sources = []
        
        try:
            # Get streaming response directly
            logger.info(f"Processing streaming query: {query_text[:100]}...")
            streaming_response = self.query_engine.query(query_text)
            
            # Store source nodes before streaming begins if available
            if hasattr(streaming_response, 'source_nodes'):
                self.last_sources = streaming_response.source_nodes
            elif hasattr(streaming_response, '_source_nodes'):
                self.last_sources = streaming_response._source_nodes
            
            # Stream the response tokens
            for text in streaming_response.response_gen:
                # Check for stop event if provided
                if stop_event and stop_event.is_set():
                    logger.info("Query streaming stopped by stop event")
                    return
                    
                yield text
                
                # Small sleep to prevent overwhelming the client
                await asyncio.sleep(0.01)
                
                # Check for stop again after yielding
                if stop_event and stop_event.is_set():
                    logger.info("Query streaming stopped by stop event after chunk")
                    return
                    
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
                    import html
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
        """Return formatted sources from the last query with enhanced metadata lookup"""
        formatted_sources = []
        seen_urls = set()
        seen_filenames = set()
        
        # Make sure metadata repository is loaded
        if not self.metadata_repository.is_loaded:
            self.metadata_repository.load_metadata_index()
        
        for node in self.last_sources:
            try:
                # Get source filename from metadata
                file_name = node.metadata.get('file_name', '')
                
                # Skip if we've already seen this file
                if file_name in seen_filenames:
                    continue
                    
                if file_name:
                    seen_filenames.add(file_name)
                    
                    # Get complete metadata from repository
                    complete_metadata = self.metadata_repository.get_metadata_by_filename(file_name)
                    
                    if complete_metadata:
                        # Use complete metadata if available
                        title = complete_metadata.get('title', 'Untitled')
                        date = complete_metadata.get('date', 'Unknown')
                        url = complete_metadata.get('url', 'No URL')
                        author = complete_metadata.get('author', 'Unknown')
                        
                        # Skip duplicate URLs
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        
                        # Get excerpt from node text
                        excerpt = self._extract_excerpt_from_node(node)
                        
                        formatted_sources.append({
                            "title": title,
                            "date": date,
                            "author": author,
                            "url": url,
                            "excerpt": excerpt
                        })
                        
                        if len(formatted_sources) >= max_sources:
                            break
                        continue
                
                # Fallback to existing method if no complete metadata found
                title = node.metadata.get('title', 'Untitled')
                date = node.metadata.get('date', 'Unknown')
                url = node.metadata.get('url', 'No URL')
                author = node.metadata.get('author', 'Unknown')
                
                # Extract from text as a last resort
                if title == 'Untitled' or date == 'Unknown' or url == 'No URL':
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
                        if in_metadata and ': ' in line:
                            key, value = line.split(': ', 1)
                            metadata[key] = value
                            metadata[key.lower()] = value
                    
                    # Use extracted metadata if found
                    title = metadata.get('Title', metadata.get('title', title))
                    date = metadata.get('Date', metadata.get('date', date))
                    url = metadata.get('URL', metadata.get('url', url))
                    author = metadata.get('Author', metadata.get('author', author))
                
                # Skip duplicate URLs
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Get excerpt
                excerpt = self._extract_excerpt_from_node(node)
                
                formatted_sources.append({
                    "title": title,
                    "date": date,
                    "author": author,
                    "url": url,
                    "excerpt": excerpt
                })
                
                if len(formatted_sources) >= max_sources:
                    break
                    
            except Exception as e:
                logger.error(f"Error formatting source: {e}")
                continue
                
        return formatted_sources
        
    def format_sources_only(self, source_nodes):
        """Format only the sources in a clean way"""
        output = ["\nSources:"]
        seen_sources = set()  # Track unique sources
        source_counter = 1  # Explicit counter for sources
        
        for node in source_nodes:
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
                excerpt = self._extract_excerpt_from_node(node)
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
            from llama_index.core import Settings
            test_prompt = "Generate a short response about communism in 1-2 sentences."
            print(f"DEBUG: Sending test prompt directly to LLM: {test_prompt}")
            response = Settings.llm.complete(test_prompt)
            print(f"DEBUG: Direct LLM response: {response}")
            return f"LLM Test Result: {response}"
        except Exception as e:
            print(f"ERROR: LLM test failed: {e}")
            import traceback
            print(traceback.format_exc())
            return f"LLM Test Error: {str(e)}"
        