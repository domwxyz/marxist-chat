import re
from datetime import datetime
from .text_utils import ensure_unicode

def clean_category(category):
    """Clean and normalize a category string"""
    if not category:
        return ""
    
    # Force to string and handle various wrapper characters
    if not isinstance(category, str):
        category = str(category)
        
    # Remove brackets, quotes, etc. from the entire string
    category = category.strip('[]\'\"')
    category = category.strip()
    
    if not category:
        return ""
    
    # Remove all special characters except letters, numbers, spaces, and hyphens
    cleaned = re.sub(r'[^\w\s-]', '', category)
    
    # Normalize spaces and hyphens
    cleaned = re.sub(r'[-\s]+', ' ', cleaned).strip()
    
    # Skip empty or single-character categories
    if len(cleaned) <= 1:
        return ""
    
    # Title case each word
    return cleaned.title()

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

def format_date(date_str):
    """Format RSS date string to YYYY-MM-DD with improved format handling"""
    if not date_str or date_str == 'Unknown Date':
        return datetime.now().strftime('%Y-%m-%d')
    
    # Try multiple date formats
    date_formats = [
        '%a, %d %b %Y %H:%M:%S %z',  # Standard RSS
        '%a, %d %b %Y %H:%M:%S %Z',  # Variant with timezone name
        '%Y-%m-%dT%H:%M:%S%z',       # ISO format with timezone
        '%Y-%m-%dT%H:%M:%SZ',        # ISO format UTC
        '%Y-%m-%d %H:%M:%S',         # Simple format
        '%d %b %Y',                  # Short format
        '%B %d, %Y'                  # Full month format
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # Fallback to regex extraction if all formats fail
    import re
    date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    if date_match:
        year, month, day = date_match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    print(f"Warning: Could not parse date '{date_str}'")
    return datetime.now().strftime('%Y-%m-%d')

def format_metadata(metadata):
    """Format metadata into a text block"""
    metadata_block = [
        "---",
        f"Title: {metadata['title']}",
        f"Date: {metadata['date']}",
        f"Author: {metadata['author']}",
        f"URL: {metadata['url']}"
    ]
    
    if metadata.get('categories'):
        categories_str = ', '.join(metadata['categories'])
        metadata_block.append(f"Categories: {categories_str}")
    
    metadata_block.append("---\n")
    return '\n'.join(metadata_block)
