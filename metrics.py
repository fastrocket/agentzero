"""Metrics evaluation functions for web components"""
from bs4 import BeautifulSoup
import re

def evaluate_layout(component) -> float:
    """Evaluate layout structure of a component"""
    # Convert Tag to BeautifulSoup if needed
    soup = BeautifulSoup(str(component), 'html.parser') if not isinstance(component, BeautifulSoup) else component
    score = 0.5  # Base score
    
    # Check semantic structure
    semantic_tags = ['header', 'nav', 'main', 'article', 'section', 'aside', 'footer']
    semantic_count = len([tag for tag in semantic_tags if soup.find(tag)])
    score += min(0.2, semantic_count * 0.05)  # Up to 0.2 for semantic tags
    
    # Check for proper heading hierarchy
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    if headings:
        score += 0.1  # Has headings
        prev_level = 0
        valid_hierarchy = True
        for h in headings:
            level = int(h.name[1])
            if level > prev_level + 1:  # Skip level
                valid_hierarchy = False
                break
            prev_level = level
        if valid_hierarchy:
            score += 0.1  # Proper hierarchy
    
    # Check for lists and tables
    if soup.find(['ul', 'ol', 'table']):
        score += 0.1  # Structured content
    
    return min(1.0, score)

def evaluate_style(component) -> float:
    """Evaluate styling completeness"""
    # Convert Tag to BeautifulSoup if needed
    soup = BeautifulSoup(str(component), 'html.parser') if not isinstance(component, BeautifulSoup) else component
    score = 0.5  # Base score
    
    # Check for CSS classes
    elements = soup.find_all(class_=True)
    if elements:
        score += min(0.2, len(elements) * 0.02)  # Up to 0.2 for styled elements
    
    # Check for custom styles
    style_tags = soup.find_all('style')
    inline_styles = soup.find_all(style=True)
    if style_tags or inline_styles:
        score += 0.1  # Has custom styles
        
    # Check for CSS variables
    styles = ' '.join([tag.string for tag in style_tags if tag.string] + [el.get('style', '') for el in inline_styles])
    if '--' in styles:  # CSS variables
        score += 0.1
        
    # Check for media queries
    if '@media' in styles:
        score += 0.1  # Responsive styles
        
    return min(1.0, score)

def evaluate_responsiveness(component) -> float:
    """Evaluate responsive design implementation"""
    # Convert Tag to BeautifulSoup if needed
    soup = BeautifulSoup(str(component), 'html.parser') if not isinstance(component, BeautifulSoup) else component
    score = 0.5  # Base score
    
    # Check for viewport meta tag
    if soup.find('meta', attrs={'name': 'viewport'}):
        score += 0.1
    
    # Check for responsive classes (Bootstrap-like)
    responsive_classes = ['container', 'row', 'col', 'sm-', 'md-', 'lg-', 'xl-']
    elements = soup.find_all(class_=True)
    responsive_count = 0
    for el in elements:
        classes = el.get('class', [])
        if any(rc in ' '.join(classes) for rc in responsive_classes):
            responsive_count += 1
    score += min(0.2, responsive_count * 0.05)  # Up to 0.2 for responsive classes
    
    # Check for media queries in styles
    style_tags = soup.find_all('style')
    styles = ' '.join([tag.string for tag in style_tags if tag.string])
    if '@media' in styles:
        score += 0.2  # Has media queries
    
    return min(1.0, score)

def evaluate_text_content(component) -> float:
    """Evaluate text content quality"""
    # Convert Tag to BeautifulSoup if needed
    soup = BeautifulSoup(str(component), 'html.parser') if not isinstance(component, BeautifulSoup) else component
    score = 0.5  # Base score
    
    # Check text length and variety
    texts = [t.strip() for t in soup.stripped_strings]
    if texts:
        # Check average text length
        avg_length = sum(len(t.split()) for t in texts) / len(texts)
        score += min(0.2, avg_length * 0.02)  # Up to 0.2 for good text length
        
        # Check text variety (unique words)
        words = set(' '.join(texts).lower().split())
        score += min(0.2, len(words) * 0.01)  # Up to 0.2 for vocabulary variety
    
    # Check for headings and paragraphs
    if soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        score += 0.1  # Has headings
    if soup.find('p'):
        score += 0.1  # Has paragraphs
    
    return min(1.0, score)

def evaluate_content_depth(component) -> float:
    """Evaluate content structure and depth"""
    # Convert Tag to BeautifulSoup if needed
    soup = BeautifulSoup(str(component), 'html.parser') if not isinstance(component, BeautifulSoup) else component
    score = 0.5  # Base score
    
    # Check content hierarchy
    heading_levels = len(set(tag.name for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])))
    score += min(0.2, heading_levels * 0.1)  # Up to 0.2 for heading hierarchy
    
    # Check content sections
    sections = len(soup.find_all(['section', 'article', 'aside']))
    score += min(0.2, sections * 0.1)  # Up to 0.2 for content sections
    
    # Check for lists and structured content
    lists = len(soup.find_all(['ul', 'ol', 'dl']))
    score += min(0.1, lists * 0.05)  # Up to 0.1 for lists
    
    return min(1.0, score)

def evaluate_media_content(component) -> float:
    """Evaluate media integration"""
    # Convert Tag to BeautifulSoup if needed
    soup = BeautifulSoup(str(component), 'html.parser') if not isinstance(component, BeautifulSoup) else component
    score = 0.5  # Base score
    
    # Check for images
    images = soup.find_all('img')
    if images:
        score += min(0.2, len(images) * 0.1)  # Up to 0.2 for images
        # Check for alt text
        if all(img.get('alt') for img in images):
            score += 0.1  # All images have alt text
    
    # Check for videos
    videos = soup.find_all(['video', 'iframe'])
    if videos:
        score += min(0.2, len(videos) * 0.1)  # Up to 0.2 for videos
    
    # Check for other media (audio, canvas, svg)
    other_media = soup.find_all(['audio', 'canvas', 'svg'])
    if other_media:
        score += min(0.1, len(other_media) * 0.05)  # Up to 0.1 for other media
    
    return min(1.0, score)
