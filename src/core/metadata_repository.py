import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core.schema import BaseNode
import logging
import re

logger = logging.getLogger("core.metadata_repository")

class MetadataRepository:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.metadata_file = cache_dir / "metadata_index.json"
        self.metadata_list = []
        self.is_loaded = False
        self.filename_index = {}  # Index for quick filename lookups
        self.feed_directory_index = {}  # Map filenames to feed directories
    
    def build_metadata_index(self, force_rebuild=False):
        """Build or rebuild the metadata index from cached documents"""
        if self.metadata_file.exists() and not force_rebuild:
            logger.info("Metadata index already exists, loading from file")
            return self.load_metadata_index()
            
        logger.info("Building metadata index from cache directory")
        self.metadata_list = []
        self.filename_index = {}
        self.feed_directory_index = {}
        
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory {self.cache_dir} does not exist")
            return False
            
        # First, check for documents directly in the cache directory (old structure)
        self._process_directory_documents(self.cache_dir, None)
            
        # Then check all subdirectories (new structure)
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                feed_name = subdir.name
                self._process_directory_documents(subdir, feed_name)
        
        # Sort by date (newest first)
        self.metadata_list.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Save to file
        self._save_metadata_index()
        
        logger.info(f"Built metadata index with {len(self.metadata_list)} entries from multiple feed sources")
        self.is_loaded = True
        return True
    
    def _process_directory_documents(self, directory: Path, feed_name: Optional[str]):
        """Process all documents in a specific directory"""
        for file_path in directory.glob("*.txt"):
            try:
                if file_path.stat().st_size == 0:
                    continue
                    
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                
                # Extract metadata section
                metadata = self._extract_metadata_from_text(content)
                
                # Add filename to metadata
                metadata["file_name"] = file_path.name
                
                # Add feed_name to metadata
                if feed_name:
                    metadata["feed_name"] = feed_name
                else:
                    # For files in the main directory, try to extract feed name from text
                    # or use "unknown" as fallback
                    embedded_feed = metadata.get("feed_name", "unknown")
                    metadata["feed_name"] = embedded_feed
                
                # Track the feed directory for this file
                self.feed_directory_index[file_path.name] = feed_name
                
                # Extract a brief excerpt for the summary
                content_parts = content.split("Content:", 1)
                if len(content_parts) > 1:
                    excerpt = content_parts[1].strip()
                    # Get first paragraph or up to 200 chars
                    excerpt_parts = excerpt.split("\n\n", 1)
                    excerpt = excerpt_parts[0].strip()
                    if len(excerpt) > 200:
                        excerpt = excerpt[:197] + "..."
                    metadata["excerpt"] = excerpt
                
                self.metadata_list.append(metadata)
                self.filename_index[file_path.name] = metadata
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    def load_metadata_index(self):
        """Load the metadata index from file"""
        if not self.metadata_file.exists():
            logger.warning("Metadata index file does not exist, build it first")
            return False
            
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata_list = json.load(f)
            
            # Build the filename index for fast lookups
            self.filename_index = {}
            self.feed_directory_index = {}
            
            for entry in self.metadata_list:
                filename = entry.get("file_name")
                feed_name = entry.get("feed_name")
                
                if filename:
                    self.filename_index[filename] = entry
                    self.feed_directory_index[filename] = feed_name
            
            logger.info(f"Loaded metadata index with {len(self.metadata_list)} entries")
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading metadata index: {e}")
            return False
    
    def _save_metadata_index(self):
        """Save the metadata index to file"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata_list, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata index: {e}")
            return False
    
    def _extract_metadata_from_text(self, text: str):
        """Extract metadata from document text"""
        metadata = {}
        lines = text.split("\n")
        
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
        
    def get_metadata_by_filename(self, filename):
        """Get complete metadata for a document by filename"""
        if not self.is_loaded:
            self.load_metadata_index()
            
        # Use the filename index for fast lookup
        return self.filename_index.get(filename)
    
    def get_file_path(self, filename):
        """Get the full file path for a document by filename"""
        if not self.is_loaded:
            self.load_metadata_index()
            
        # Get the feed directory for this file
        feed_dir = self.feed_directory_index.get(filename)
        
        if feed_dir:
            # File is in a feed subdirectory
            return self.cache_dir / feed_dir / filename
        else:
            # File is directly in the cache directory
            return self.cache_dir / filename
    
    def get_metadata_by_feed(self, feed_name):
        """Get all metadata entries for a specific feed"""
        if not self.is_loaded:
            self.load_metadata_index()
            
        return [entry for entry in self.metadata_list if entry.get("feed_name") == feed_name]
    
    def get_all_feed_names(self):
        """Get a list of all feed names in the metadata"""
        if not self.is_loaded:
            self.load_metadata_index()
            
        feed_names = set()
        for entry in self.metadata_list:
            feed_name = entry.get("feed_name")
            if feed_name:
                feed_names.add(feed_name)
                
        return list(feed_names)
    
    def get_formatted_context(self, max_entries=100, max_chars=4000):
        """Get a formatted context string for the LLM with document metadata"""
        if not self.is_loaded:
            self.load_metadata_index()
            
        if not self.metadata_list:
            return "No document metadata available."
            
        # Limit to max_entries
        entries_to_use = self.metadata_list[:max_entries]
        
        # Build the context string
        context_parts = ["DOCUMENT INDEX:"]
        
        for i, entry in enumerate(entries_to_use, 1):
            feed_name = entry.get('feed_name', 'Unknown')
            entry_text = f"{i}. [{feed_name}] {entry.get('title', 'Untitled')} ({entry.get('date', 'Unknown')})"
            author = entry.get('author', '')
            if author and author != 'Unknown' and author != 'Unknown Author':
                entry_text += f" by {author}"
                
            # Add categories if available
            categories = entry.get('categories', '')
            if categories:
                entry_text += f" - Categories: {categories}"
                
            context_parts.append(entry_text)
            
            # Add excerpt if available
            excerpt = entry.get('excerpt', '')
            if excerpt:
                context_parts.append(f"   Summary: {excerpt}")
            
            # Check if we're approaching the character limit
            current_length = sum(len(part) for part in context_parts)
            if current_length > max_chars:
                context_parts.append(f"... (showing {i} of {len(self.metadata_list)} documents)")
                break
        
        return "\n".join(context_parts)
        