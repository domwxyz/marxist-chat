import time
import feedparser
from datetime import datetime
from pathlib import Path
from llama_index.core import Document
from typing import List, Dict, Any

from utils.text_utils import ensure_unicode, preprocess_content, clean_rss_boilerplate
from utils.metadata_utils import clean_category, sanitize_filename, format_date, format_metadata
import config

class FeedProcessor:
    def __init__(self, feed_urls=None, cache_dir=None):
        self.feed_urls = feed_urls or config.RSS_FEED_URLS
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        
    def fetch_all_feeds(self):
        """Fetch and process all configured RSS feeds"""
        all_entries = []
        for feed_url in self.feed_urls:
            print(f"Processing {feed_url}")
            entries = self.fetch_rss_entries(feed_url)
            all_entries.extend(entries)
        
        print(f"Found {len(all_entries)} total entries")
        return all_entries
    
    def fetch_rss_entries(self, feed_url):
        """Fetch all entries from WordPress RSS feed with pagination - optimized version"""
        entries = []
        page = 1
        has_more = True
        seen_urls = set()  # Use a set for faster duplicate checking
        
        print(f"Fetching articles from {feed_url}")
        
        while has_more:
            # Handle WordPress pagination pattern
            current_url = f"{feed_url.rstrip('/')}/?paged={page}" if page > 1 else feed_url
            
            # Add a user agent to avoid being blocked
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; RSSBot/1.0)'}
            feed = feedparser.parse(current_url, request_headers=headers)
            
            print(f"Processing page {page}... Found {len(feed.entries)} entries.")
            
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
            
            print(f"Added {new_entries} new entries from page {page}.")
                
            # WordPress pagination fallback logic
            if new_entries == 0:
                print("No new entries found on this page.")
                has_more = False
            else:
                page += 1
                print(f"Moving to page {page}...")

            # Check for standard RSS pagination links
            next_page = None
            for link in feed.feed.get("links", []):
                if link.rel == "next":
                    next_page = link.href
                    print(f"Found 'next' link: {next_page}")
                    break
            
            if next_page and next_page != current_url:
                print(f"Switching to next URL: {next_page}")
                feed_url = next_page  # Update base URL if different
                page = 1  # Reset page counter
                
            time.sleep(0.2)  # Respect server resources
        
        print(f"Finished fetching all pages. Total entries: {len(entries)}")
        return entries
    
    def extract_metadata_from_entry(self, entry):
        """Extract and format metadata from a feedparser entry - optimized version"""
        # Extract basic metadata - avoid repeated dictionary lookups
        title = entry.get('title', 'Untitled')
        published_date = entry.get('published', entry.get('pubDate', 'Unknown Date'))
        author = entry.get('author', 'Unknown Author')
        url = entry.get('link', 'No URL')
        
        # Format date only once
        formatted_date = format_date(published_date)
        
        # Process categories more efficiently
        categories = []
        category_set = set()  # Use a set to avoid duplicates more efficiently
        
        # Process tags field (only if it exists)
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
        
        # Process category field, handling both single and multiple categories
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
        
        # Return metadata directly - more efficient than building dictionary incrementally
        return {
            "title": title,
            "date": formatted_date,
            "author": author,
            "url": url,
            "categories": categories
        }
    
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
            
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        documents = []
        last_known_date = datetime.now().strftime('%Y-%m-%d')  # Default fallback
        
        # Pre-compute existing filenames to avoid repeated directory checks
        existing_filenames = set(f.name for f in self.cache_dir.glob("*.txt"))
        
        print(f"Processing {len(entries)} entries...")
        
        # Process entries in batches for improved performance
        batch_size = 50
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i+batch_size]
            batch_documents = []
            
            for entry in batch:
                try:
                    # Extract metadata from the full feedparser entry - more efficient
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
                    
                    # Check against pre-computed set - much faster than checking file existence repeatedly
                    counter = 1
                    original_filename = filename
                    while filename in existing_filenames:
                        filename = f"{date_prefix}_{safe_title}-{counter}.txt"
                        counter += 1
                    
                    # Add to our tracking set for future checks in the same batch
                    existing_filenames.add(filename)
                    
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
                    batch_documents.append(doc)
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
        for doc in documents:
            try:
                filepath = self.cache_dir / doc.metadata["file_name"]
                with open(filepath, "w", encoding="utf-8", errors="replace") as f:
                    f.write(doc.text)
            except Exception as e:
                print(f"Error writing document {doc.metadata.get('file_name', 'unknown')}: {str(e)}")
                continue
