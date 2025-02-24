from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import List, Dict, Any

import config
from core.vector_store import VectorStoreManager
from core.llm_manager import LLMManager

class QueryEngine:
    def __init__(self, vector_store_dir=None):
        self.vector_store_manager = VectorStoreManager(vector_store_dir)
        self.query_engine = None
        self.index = None
        self.last_sources = []
    
    def initialize(self):
        """Initialize the query engine with the vector store"""
        self.index = self.vector_store_manager.load_vector_store()
        if not self.index:
            return False
        
        # Create query template
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
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ],
            text_qa_template=query_template,
            streaming=True  # Enable streaming for future WebSocket use
        )
        
        return True
    
    def query(self, query_text):
        """Process a query and return the response"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call initialize() first.")
        
        # Reset sources for the new query
        self.last_sources = []
        
        # Get response
        response = self.query_engine.query(query_text)
        
        # Store sources for later retrieval
        if hasattr(response, 'source_nodes'):
            self.last_sources = response.source_nodes
            
        return response
    
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
                    print(f"Error formatting source {idx}: {str(e)}")
                    continue
        
        return "\n".join(output)
    
    def get_formatted_sources(self):
        """Return formatted sources from the last query"""
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
                    "url": url,
                    "excerpt": excerpt
                })
                
                # Limit to 5 sources
                if len(formatted_sources) >= 5:
                    break
                    
            except Exception as e:
                print(f"Error formatting source: {e}")
                continue
                
        return formatted_sources