import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("core.metadata_repository")

class MetadataRepository:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.metadata_file = cache_dir / "metadata_index.json"
        self.metadata_list = []
        self.is_loaded = False
    
    def build_metadata_index(self, force_rebuild=False):
        """Build or rebuild the metadata index from cached documents"""
        if self.metadata_file.exists() and not force_rebuild:
            logger.info("Metadata index already exists, loading from file")
            return self.load_metadata_index()
            
        logger.info("Building metadata index from cache directory")
        self.metadata_list = []
        
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory {self.cache_dir} does not exist")
            return False
            
        # Process all document files in the cache directory
        for file_path in self.cache_dir.glob("*.txt"):
            try:
                if file_path.stat().st_size == 0:
                    continue
                    
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                
                # Extract metadata section
                metadata = self._extract_metadata_from_text(content)
                
                # Add filename to metadata
                metadata["file_name"] = file_path.name
                
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
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        # Sort by date (newest first)
        self.metadata_list.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Save to file
        self._save_metadata_index()
        
        logger.info(f"Built metadata index with {len(self.metadata_list)} entries")
        self.is_loaded = True
        return True
    
    def load_metadata_index(self):
        """Load the metadata index from file"""
        if not self.metadata_file.exists():
            logger.warning("Metadata index file does not exist, build it first")
            return False
            
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata_list = json.load(f)
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
            entry_text = f"{i}. {entry.get('title', 'Untitled')} ({entry.get('date', 'Unknown')})"
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