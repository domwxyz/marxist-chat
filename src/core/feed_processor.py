import time
import feedparser
from datetime import datetime
from pathlib import Path
from llama_index.core import Document
from typing import List, Dict, Any, Optional
import os
import re

from utils.text_utils import ensure_unicode, preprocess_content, clean_rss_boilerplate
from utils.metadata_utils import clean_category, sanitize_filename, format_date, format_metadata
import config

class FeedProcessor:
    def __init__(self, feed_urls=None, cache_dir=None):
        # If feed_urls is provided, use it; otherwise, use URLs from config
        if feed_urls:
            self.feed_urls = feed_urls
            # Create a simple config for each provided URL (assuming standard pagination)
            self.feed_configs = {url: {"url": url, "pagination_type": "standard"} for url in feed_urls}
        else:
            # Use the URLs and configurations from config.RSS_FEED_CONFIG
            self.feed_urls = [feed_config["url"] for feed_config in config.RSS_FEED_CONFIG]
            self.feed_configs = self._build_feed_config_map()
            
        self.base_cache_dir = cache_dir or config.CACHE_DIR
        self.base_cache_dir.mkdir(exist_ok=True)
    
    def _build_feed_config_map(self):
        """Build a mapping of feed URLs to their configuration"""
        feed_config_map = {}
        for feed_config in config.RSS_FEED_CONFIG:
            feed_config_map[feed_config["url"]] = feed_config
        return feed_config_map
        
    def _get_feed_directory_name(self, feed_url: str) -> str:
        """Convert a feed URL to a safe directory name"""
        # Extract domain from URL
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', feed_url)
        if domain_match:
            domain = domain_match.group(1)
            # Replace dots and slashes with hyphens
            return domain.replace('.', '-').replace('/', '-')
        else:
            # Fallback to a sanitized version of the URL
            sanitized = re.sub(r'[^\w\-]', '-', feed_url)
            return sanitized[:50]  # Limit length
    
    def _get_feed_cache_dir(self, feed_url: str) -> Path:
        """Get the cache directory for a specific feed"""
        feed_dir_name = self._get_feed_directory_name(feed_url)
        feed_cache_dir = self.base_cache_dir / feed_dir_name
        feed_cache_dir.mkdir(exist_ok=True)
        return feed_cache_dir
        
    def fetch_all_feeds(self):
        """Fetch and process all configured RSS feeds"""
        all_entries = []
        for feed_url in self.feed_urls:
            print(f"Processing {feed_url}")
            feed_entries = self.fetch_rss_entries(feed_url)
            # Add source feed information to each entry
            for entry in feed_entries:
                entry['_feed_url'] = feed_url  # Add source feed URL for tracking
            all_entries.extend(feed_entries)
        
        print(f"Found {len(all_entries)} total entries")
        return all_entries
    
    def fetch_rss_entries(self, feed_url):
        """Fetch all entries from RSS feed with pagination support for different CMS types"""
        entries = []
        seen_urls = set()  # Use a set for faster duplicate checking
        
        print(f"Fetching articles from {feed_url}")
        
        # Get feed configuration
        feed_config = self.feed_configs.get(feed_url, {"pagination_type": "standard"})
        pagination_type = feed_config.get("pagination_type", "standard")
        
        # Initialize pagination variables
        page = 1  # For WordPress
        limitstart = 0  # For Joomla
        limit_increment = feed_config.get("limit_increment", 5)  # Default increment for Joomla feeds
        has_more = True
        
        while has_more:
            # Handle pagination based on feed type
            if pagination_type == "joomla":
                # Joomla pagination with limitstart
                if "format=feed" in feed_url:
                    base_url = feed_url.split("?")[0]
                    current_url = f"{base_url}?format=feed&limitstart={limitstart}"
                else:
                    # Handle case where feed_url doesn't already have format=feed
                    if "?" in feed_url:
                        current_url = f"{feed_url}&format=feed&limitstart={limitstart}"
                    else:
                        current_url = f"{feed_url}?format=feed&limitstart={limitstart}"
                        
                pagination_desc = f"limitstart={limitstart}"
            elif pagination_type == "wordpress":
                # WordPress pagination
                current_url = f"{feed_url.rstrip('/')}/?paged={page}" if page > 1 else feed_url
                pagination_desc = f"page {page}"
            else:
                # Standard/unknown pagination - just use the URL as is
                current_url = feed_url
                pagination_desc = "first page"
                # For unknown pagination types, we'll only process the first page
                has_more = False
            
            print(f"Fetching from URL: {current_url}")
            
            # Add a user agent to avoid being blocked
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; RSSBot/1.0)'}
            feed = feedparser.parse(current_url, request_headers=headers)
            
            print(f"Processing {pagination_desc}... Found {len(feed.entries)} entries.")
            
            if not feed.entries:
                print("No entries found on this page. Moving to next URL if available.")
                has_more = False
                break
                
            new_entries = 0
            
            # Process entries in a batch
            for entry in feed.entries:
                entry_url = entry.get('link', '')
                
                # Skip duplicates using set lookup (much faster than list)
                if not entry_url or entry_url in seen_urls:
                    continue
                    
                # Store URL in the seen set
                seen_urls.add(entry_url)
                entries.append(entry)
                new_entries += 1
            
            print(f"Added {new_entries} new entries.")
                
            # Update pagination based on feed type
            if new_entries == 0:
                print("No new entries found on this page.")
                has_more = False
            else:
                if pagination_type == "joomla":
                    # For Joomla, increment the limitstart parameter
                    limitstart += limit_increment
                    print(f"Moving to limitstart={limitstart}...")
                elif pagination_type == "wordpress":
                    # For WordPress, increment the page number
                    page += 1
                    print(f"Moving to page {page}...")

            # Check for standard RSS pagination links (works with some feeds)
            next_page = None
            for link in feed.feed.get("links", []):
                if link.rel == "next":
                    next_page = link.href
                    print(f"Found 'next' link: {next_page}")
                    break
            
            if next_page and next_page != current_url:
                print(f"Switching to next URL: {next_page}")
                feed_url = next_page  # Update base URL if different
                if pagination_type == "joomla":
                    # Reset limitstart if we're switching to a completely different URL
                    limitstart = 0
                elif pagination_type == "wordpress":
                    # Reset page counter if we're switching to a different URL
                    page = 1
                    
            time.sleep(0.2)  # Respect server resources
        
        print(f"Finished fetching all pages. Total entries: {len(entries)}")
        return entries
    
    def fetch_new_entries(self, since_date=None):
        """Fetch only new entries from RSS feeds since a given date"""
        all_entries = []
        
        if not since_date:
            print("No date specified, fetching all entries.")
            return self.fetch_all_feeds()
            
        print(f"Fetching entries newer than {since_date}")
        
        for feed_url in self.feed_urls:
            print(f"Processing {feed_url} for new content")
            entries = []
            
            # Get feed configuration
            feed_config = self.feed_configs.get(feed_url, {"pagination_type": "standard"})
            pagination_type = feed_config.get("pagination_type", "standard")
            
            # Initialize pagination variables
            page = 1  # For WordPress
            limitstart = 0  # For Joomla
            limit_increment = feed_config.get("limit_increment", 5)  # Default increment for Joomla feeds
            has_more = True
            seen_urls = set()
            found_article_count = 0
            
            while has_more:
                # Handle pagination based on feed type
                if pagination_type == "joomla":
                    # Joomla pagination with limitstart
                    if "format=feed" in feed_url:
                        base_url = feed_url.split("?")[0]
                        current_url = f"{base_url}?format=feed&limitstart={limitstart}"
                    else:
                        # Handle case where feed_url doesn't already have format=feed
                        if "?" in feed_url:
                            current_url = f"{feed_url}&format=feed&limitstart={limitstart}"
                        else:
                            current_url = f"{feed_url}?format=feed&limitstart={limitstart}"
                    pagination_desc = f"limitstart={limitstart}"
                elif pagination_type == "wordpress":
                    # WordPress pagination
                    current_url = f"{feed_url.rstrip('/')}/?paged={page}" if page > 1 else feed_url
                    pagination_desc = f"page {page}"
                else:
                    # Standard/unknown pagination - just use the URL as is
                    current_url = feed_url
                    pagination_desc = "first page"
                    
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; RSSBot/1.0)'}
                feed = feedparser.parse(current_url, request_headers=headers)
                
                print(f"Processing {pagination_desc}... Found {len(feed.entries)} entries.")
                
                if not feed.entries:
                    print("No entries found on this page.")
                    has_more = False
                    break
                    
                new_entries_on_page = 0
                
                for entry in feed.entries:
                    entry_url = entry.get('link', '')
                    
                    if not entry_url or entry_url in seen_urls:
                        continue
                        
                    # Extract date to check if it's newer than since_date
                    published_date = entry.get('published', entry.get('pubDate', 'Unknown Date'))
                    entry_date = format_date(published_date)
                    
                    # Skip if this entry is older than or equal to our cutoff date
                    if entry_date <= since_date:
                        print(f"Found entry from {entry_date}, which is not newer than {since_date}")
                        # If we start seeing older articles, we can stop
                        found_article_count += 1
                        if found_article_count >= 3:  # Stop after finding a few older articles
                            has_more = False
                            break
                        continue
                    
                    # Add feed_url to entry for source tracking
                    entry['_feed_url'] = feed_url
                    
                    seen_urls.add(entry_url)
                    entries.append(entry)
                    all_entries.append(entry)
                    new_entries_on_page += 1
                
                print(f"Added {new_entries_on_page} new entries.")
                    
                # If no new entries on this page or we've found older content, stop
                if new_entries_on_page == 0 or not has_more:
                    has_more = False
                else:
                    # Update pagination variables based on the feed type
                    if pagination_type == "joomla":
                        limitstart += limit_increment
                        print(f"Moving to limitstart={limitstart}...")
                    elif pagination_type == "wordpress":
                        page += 1
                        print(f"Moving to page {page}...")
                    
                # Check for standard RSS pagination links
                next_page = None
                for link in feed.feed.get("links", []):
                    if link.rel == "next":
                        next_page = link.href
                        break
                
                if next_page and next_page != current_url:
                    print(f"Found 'next' link: {next_page}")
                    feed_url = next_page  # Update base URL
                    
                    # Reset pagination counters when changing URLs
                    if pagination_type == "joomla":
                        limitstart = 0
                    elif pagination_type == "wordpress":
                        page = 1
                    
                time.sleep(0.2)  # Respect server resources
        
        print(f"Finished fetching new entries. Total: {len(all_entries)}")
        return all_entries
    
    def extract_metadata_from_entry(self, entry):
        """Extract and format metadata with improved feed-specific handling"""
        # Base metadata dictionary
        metadata = {}
        
        # Extract feed source
        feed_url = entry.get('_feed_url', 'Unknown Feed')
        is_joomla = 'marxist.com' in feed_url
        
        # Title extraction (consistent across feeds)
        metadata['title'] = entry.get('title', 'Untitled')
        
        # Date extraction and standardization
        published_date = entry.get('published', entry.get('pubDate', 'Unknown Date'))
        metadata['date'] = format_date(published_date)
        
        # URL extraction (consistent across feeds)
        metadata['url'] = entry.get('link', 'No URL')
        
        # Feed tracking
        metadata['feed_url'] = feed_url
        metadata['feed_name'] = self._get_feed_directory_name(feed_url)
        
        # Author extraction with feed-specific handling
        if is_joomla:
            # Joomla-specific author extraction logic
            if hasattr(entry, 'dc_creator'):
                metadata['author'] = getattr(entry, 'dc_creator', 'Unknown Author')
            elif hasattr(entry, 'creator'):
                metadata['author'] = getattr(entry, 'creator', 'Unknown Author')
            elif hasattr(entry, 'author_detail') and hasattr(entry.author_detail, 'name'):
                metadata['author'] = entry.author_detail.name
            else:
                metadata['author'] = entry.get('author', 'Unknown Author')
        else:
            # WordPress author extraction
            metadata['author'] = entry.get('author', 'Unknown Author')
        
        # Extract and clean categories
        metadata['categories'] = self._extract_categories(entry)
        
        return metadata

    def _extract_categories(self, entry):
        """Extract and clean categories with deduplication"""
        categories = []
        category_set = set()  # For deduplication
        
        # Process tags field
        if hasattr(entry, 'tags'):
            for tag in entry.tags:
                # Handle dictionary-style tags
                tag_text = tag.get('term', '') if isinstance(tag, dict) else str(tag)
                
                # Process multi-category tags
                if ',' in tag_text:
                    for raw_cat in tag_text.split(','):
                        cleaned = clean_category(raw_cat)
                        if cleaned and cleaned not in category_set:
                            category_set.add(cleaned)
                            categories.append(cleaned)
                else:
                    cleaned = clean_category(tag_text)
                    if cleaned and cleaned not in category_set:
                        category_set.add(cleaned)
                        categories.append(cleaned)
        
        # Process category field
        if hasattr(entry, 'category'):
            # Handle string or list
            if isinstance(entry.category, str):
                cleaned = clean_category(entry.category)
                if cleaned and cleaned not in category_set:
                    categories.append(cleaned)
            elif isinstance(entry.category, list):
                for cat in entry.category:
                    cleaned = clean_category(cat)
                    if cleaned and cleaned not in category_set:
                        categories.append(cleaned)
        
        # Additional category formats for Joomla
        if hasattr(entry, 'categories'):
            for cat in entry.categories:
                cleaned = clean_category(cat)
                if cleaned and cleaned not in category_set:
                    categories.append(cleaned)
        
        return categories
    
    def extract_content_sections(self, entry):
        """Extract and clean the description and main content - optimized version"""
        # Get description from summary field - only process if it exists
        description = ""
        if hasattr(entry, 'summary') and entry.summary:
            description = ensure_unicode(entry.summary)
            description = clean_rss_boilerplate(description)
            description = preprocess_content(description)
        
        # Get main content from content field - only process if it exists
        content = ""
        if hasattr(entry, 'content') and entry.content:
            # Check if content exists and has value field to avoid unnecessary processing
            if entry.content[0].get('value'):
                content = ensure_unicode(entry.content[0].get('value', ''))
                content = clean_rss_boilerplate(content)
                content = preprocess_content(content)
        
        return description, content
    
    def process_entries(self, entries):
        """Process and store entries as documents - optimized version"""
        if not entries:
            print("No entries to process.")
            return []
            
        documents = []
        
        # Generate a list of existing files for each feed source
        existing_files_by_feed = {}
        for feed_url in self.feed_urls:
            feed_cache_dir = self._get_feed_cache_dir(feed_url)
            existing_files_by_feed[feed_url] = set(f.name for f in feed_cache_dir.glob("*.txt"))
        
        print(f"Processing {len(entries)} entries...")
        
        # Process entries in batches for improved performance
        batch_size = 50
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i+batch_size]
            batch_documents = []
            
            for entry in batch:
                try:
                    # Get the source feed URL (added during fetching)
                    feed_url = entry.get('_feed_url', self.feed_urls[0])  # Default to first feed if not found
                    
                    # Get the cache directory for this feed
                    feed_cache_dir = self._get_feed_cache_dir(feed_url)
                    
                    # Track last known date per feed
                    last_known_date = datetime.now().strftime('%Y-%m-%d')  # Default fallback
                    
                    # Extract metadata from the full feedparser entry
                    metadata = self.extract_metadata_from_entry(entry)
                    
                    # If we got a valid date, update our last_known_date
                    if metadata['date'] and metadata['date'] != 'Unknown Date':
                        last_known_date = metadata['date']
                    else:
                        # Use the last known date if this entry's date is missing/unknown
                        metadata['date'] = last_known_date
                    
                    # Generate safe filename
                    date_prefix = metadata['date']
                    safe_title = sanitize_filename(metadata['title'])
                    filename = f"{date_prefix}_{safe_title}.txt"
                    
                    # Check against existing files for this feed
                    counter = 1
                    original_filename = filename
                    while filename in existing_files_by_feed.get(feed_url, set()):
                        filename = f"{date_prefix}_{safe_title}-{counter}.txt"
                        counter += 1
                    
                    # Add to our tracking set for future checks in the same batch
                    existing_files_by_feed.setdefault(feed_url, set()).add(filename)
                    
                    # Add filename to metadata
                    metadata['file_name'] = filename
                    
                    # Extract content sections with explicit encoding handling
                    description, content = self.extract_content_sections(entry)
                    
                    # Create metadata text block
                    metadata_formatted = format_metadata(metadata)
                    
                    # Combine content with clear section markers - build as list and join once
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
                        metadata=metadata
                    )
                    batch_documents.append((doc, feed_url))
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"Error processing entry {entry.get('title', 'unknown')}: {str(e)}")
                    continue
            
            # Process writing files in a batch
            self._write_documents_batch(batch_documents)
            print(f"Processed batch of {len(batch_documents)} documents. Total: {len(documents)}")
        
        print(f"Successfully processed {len(documents)} documents.")
        return documents

    def _write_documents_batch(self, documents):
        """Write a batch of documents to files - optimized file writing"""
        for doc, feed_url in documents:
            try:
                # Use the feed-specific cache directory
                feed_cache_dir = self._get_feed_cache_dir(feed_url)
                filepath = feed_cache_dir / doc.metadata["file_name"]
                
                with open(filepath, "w", encoding="utf-8", errors="replace") as f:
                    f.write(doc.text)
            except Exception as e:
                print(f"Error writing document {doc.metadata.get('file_name', 'unknown')}: {str(e)}")
                continue
                