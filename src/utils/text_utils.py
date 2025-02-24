import re
import chardet
import unicodedata

def ensure_unicode(text):
    """Convert various text formats to normalized Unicode"""
    if text is None:
        return ""
        
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except UnicodeDecodeError:
            try:
                raw_data = text if text else b''
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] if detected and detected['encoding'] else 'utf-8'
                return text.decode(encoding, errors='replace')
            except Exception:
                return text.decode('utf-8', errors='replace')
    
    if isinstance(text, str):
        return unicodedata.normalize('NFKC', text)
        
    return str(text)

def preprocess_content(content):
    """Clean and normalize content while preserving Unicode"""
    if not content:
        return ""
    
    # Ensure content is Unicode
    content = ensure_unicode(content)
    
    # Remove script and style elements
    content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
    content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
    
    # Handle WordPress image captions
    content = re.sub(r'<div[^>]*class="wp-caption[^>]*>.*?</div>', '', content, flags=re.DOTALL)
    
    # Convert HTML line breaks to newlines
    content = content.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    content = content.replace('</p>', '\n\n').replace('<p>', '')
    
    # Remove remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Fix common HTML entities with Unicode equivalents
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&mdash;': '—',
        '&ndash;': '–',
        '&hellip;': '…',
        '&#8217;': ''',
        '&#8216;': ''',
        '&#8220;': '"',
        '&#8221;': '"',
    }
    for entity, char in html_entities.items():
        content = content.replace(entity, char)
    
    # Normalize whitespace while preserving paragraphs
    content = re.sub(r'\s+([.,!?])', r'\1', content)
    content = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Normalize Unicode forms
    content = unicodedata.normalize('NFKC', content)
    
    # Clean up lines while preserving structure
    content = '\n'.join(line.strip() for line in content.splitlines())
    
    return content.strip()

def clean_rss_boilerplate(content):
    """Remove RSS feed boilerplate text and links"""
    if not content:
        return ""
    
    # Handle common RSS boilerplate patterns
    patterns = [
        r'The post.*?appeared first on.*?\.',
        r'<a href="https://communistusa\.org[^>]*>.*?</a>',
        r'org">Revolutionary Communists of America\.',
        r'org">[^<]*</a>',
        r'\[ ?to read the full analysis[^\]]*\]',
        r'\[ ?read more[^\]]*\]',
        r'Read more\.\.\..*$',
        r'Continue reading.*$'
    ]
    
    # Apply each cleanup pattern
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up any resulting blank lines from removed content
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Remove extra whitespace while preserving paragraph structure
    lines = [line.strip() for line in content.splitlines()]
    content = '\n'.join(line for line in lines if line)
    
    return content.strip()
