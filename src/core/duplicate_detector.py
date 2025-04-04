import re
import difflib
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("core.duplicate_detector")

class DuplicateDetector:
    """
    Detects duplicate articles across different sources based primarily on title similarity,
    publication dates, and republishing markers.
    """

    def __init__(self, metadata_repository):
        """Initialize the duplicate detector with metadata repository"""
        self.metadata_repository = metadata_repository
        self.title_cache = {}  # normalized title -> metadata
        self.url_cache = set()  # Set of URLs already processed
        self.max_date_diff = timedelta(days=7)  # Max difference between publication dates
        self.title_similarity_threshold = 0.9  # Minimum similarity for titles to be considered duplicates
        
        # Load existing content
        self._initialize_caches()
        
    def _initialize_caches(self):
        """Initialize caches from metadata repository"""
        if not self.metadata_repository.is_loaded:
            self.metadata_repository.load_metadata_index()
            
        # Process all existing documents
        for metadata in self.metadata_repository.metadata_list:
            # Cache URL
            url = metadata.get('url')
            if url and url != 'No URL':
                self.url_cache.add(url.lower())
                
            # Cache normalized title
            title = metadata.get('title')
            if title:
                normalized_title = self._normalize_title(title)
                self.title_cache[normalized_title] = metadata
                
        logger.info(f"Initialized duplicate detector with {len(self.url_cache)} URLs and {len(self.title_cache)} titles")
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize a title for comparison by removing special chars and lowercasing.
        """
        if not title:
            return ""
            
        # Convert to lowercase
        normalized = title.lower()
        
        # Remove punctuation and special characters
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Optionally remove common article words
        normalized = re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|with|by)\b', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def is_duplicate_url(self, url: str) -> bool:
        """Check if a URL has already been processed"""
        if not url:
            return False
        return url.lower() in self.url_cache
    
    def find_similar_titles(self, title: str) -> List[Dict]:
        """Find metadata entries with similar titles"""
        if not title:
            return []
            
        normalized_title = self._normalize_title(title)
        similar_titles = []
        
        # Check each title in cache
        for existing_title, metadata in self.title_cache.items():
            # Don't compare very short titles
            if len(normalized_title) < 5 or len(existing_title) < 5:
                continue
                
            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, normalized_title, existing_title).ratio()
            if similarity >= self.title_similarity_threshold:
                similar_titles.append((metadata, similarity))
                
        # Sort by similarity (highest first)
        similar_titles.sort(key=lambda x: x[1], reverse=True)
        return [meta for meta, sim in similar_titles]
    
    def is_duplicate(self, metadata: Dict[str, str], content: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if an article is a duplicate based on title similarity, date proximity,
        and republishing markers.
        
        Returns:
            Tuple[bool, Optional[Dict]]: (is_duplicate, original_metadata)
        """
        # Check for republish markers in content
        is_republished = self._has_republish_markers(content)
        
        # Find similar titles
        title = metadata.get('title', '')
        similar_titles = self.find_similar_titles(title)
        
        # If no similar titles found, not a duplicate
        if not similar_titles:
            return False, None
        
        # Extract publication date from new metadata
        new_date_str = metadata.get('date', '')
        if not new_date_str:
            new_date_str = datetime.now().strftime('%Y-%m-%d')
        
        try:
            new_date = datetime.strptime(new_date_str, '%Y-%m-%d')
        except ValueError:
            # If date parsing fails, use current date
            new_date = datetime.now()
        
        # Check each similar title to find potential duplicates
        for similar_meta in similar_titles:
            # Extract publication date
            existing_date_str = similar_meta.get('date', '')
            try:
                existing_date = datetime.strptime(existing_date_str, '%Y-%m-%d')
            except ValueError:
                # Skip if date parsing fails
                continue
                
            # Calculate date difference
            date_diff = abs(new_date - existing_date)
            
            # If published within max_date_diff and has republish markers, it's a duplicate
            if date_diff <= self.max_date_diff:
                if is_republished:
                    logger.info(f"Found republished article: '{title}' with marker, similar to '{similar_meta.get('title')}'")
                    return self._determine_original(metadata, similar_meta)
                
                # No republish markers, but very similar title and close date - likely duplicate
                logger.info(f"Found likely duplicate: '{title}' with date {new_date_str}, similar to '{similar_meta.get('title')}' with date {existing_date_str}")
                return self._determine_original(metadata, similar_meta)
        
        # No match found
        return False, None
    
    def _has_republish_markers(self, content: str) -> bool:
        """Check for markers indicating republished content"""
        markers = [
            r'[oO]riginally published',
            r'[rR]epublished from',
            r'[fF]irst published',
            r'[pP]ublished on',
            r'[rR]epublished with',
            r'[rR]eposted from',
            r'[cC]ross-?posted from'
        ]
        
        pattern = '|'.join(markers)
        return bool(re.search(pattern, content))
    
    def _determine_original(self, meta1: Dict[str, str], meta2: Dict[str, str]) -> Tuple[bool, Dict]:
        """
        Determine which article is the original based on publication date
        Returns (is_duplicate, original_metadata)
        """
        # Extract dates
        date1 = meta1.get('date', '')
        date2 = meta2.get('date', '')
        
        try:
            date1_obj = datetime.strptime(date1, '%Y-%m-%d') if date1 else datetime.now()
            date2_obj = datetime.strptime(date2, '%Y-%m-%d') if date2 else datetime.now()
            
            # Earlier date is original
            if date1_obj < date2_obj:
                return True, meta1  # meta1 is original
            else:
                return True, meta2  # meta2 is original
                
        except ValueError:
            # If date parsing fails, use string comparison
            if date1 < date2:
                return True, meta1
            else:
                return True, meta2
    
    def add_document(self, metadata: Dict[str, str]):
        """Add a document to the detection system"""
        # Add URL to cache
        url = metadata.get('url')
        if url and url != 'No URL':
            self.url_cache.add(url.lower())
            
        # Add title to cache
        title = metadata.get('title')
        if title:
            normalized_title = self._normalize_title(title)
            self.title_cache[normalized_title] = metadata
            