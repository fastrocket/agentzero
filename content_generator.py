import requests
from bs4 import BeautifulSoup
import json
import random
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self):
        self.image_sources = [
            "https://unsplash.com/search/photos/",  # Free high-quality photos
            "https://www.pexels.com/search/",       # Free stock photos
            "https://pixabay.com/images/search/"    # Free images and royalty-free stock
        ]
        
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search the web using DuckDuckGo (no rate limits)"""
        try:
            # Use DuckDuckGo HTML search
            encoded_query = quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.select('.result')[:num_results]:
                title = result.select_one('.result__title')
                snippet = result.select_one('.result__snippet')
                link = result.select_one('.result__url')
                
                if title and snippet and link:
                    results.append({
                        'title': title.get_text(strip=True),
                        'snippet': snippet.get_text(strip=True),
                        'url': link.get_text(strip=True)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return []
    
    def get_free_image(self, query: str) -> Optional[str]:
        """Get a free image URL from various sources"""
        try:
            # Randomly select an image source
            source = random.choice(self.image_sources)
            encoded_query = quote_plus(query)
            url = f"{source}{encoded_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Different selectors for different sources
            if 'unsplash.com' in url:
                img = soup.select_one('figure img[srcset]')
                if img and img.get('srcset'):
                    return img['srcset'].split(',')[0].split(' ')[0]
            
            elif 'pexels.com' in url:
                img = soup.select_one('img.photo-item__img')
                if img and img.get('src'):
                    return img['src']
            
            elif 'pixabay.com' in url:
                img = soup.select_one('div.item img')
                if img and img.get('src'):
                    return img['src']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting free image: {e}")
            return None
    
    def generate_article(self, topic: str) -> Tuple[str, List[str]]:
        """Generate an article with images based on web search"""
        try:
            # Search for content
            results = self.search_web(topic, num_results=3)
            if not results:
                return "", []
            
            # Generate article structure
            article_parts = []
            image_urls = []
            
            # Add title
            article_parts.append(f"<h1>{results[0]['title']}</h1>")
            
            # Try to get a header image
            header_image = self.get_free_image(topic)
            if header_image:
                article_parts.append(f'<img src="{header_image}" alt="{topic}" class="header-image">')
                image_urls.append(header_image)
            
            # Generate content from search results
            for i, result in enumerate(results):
                # Add subheading
                article_parts.append(f"<h2>{result['title']}</h2>")
                
                # Add content paragraph
                article_parts.append(f"<p>{result['snippet']}</p>")
                
                # Try to add an image for each section
                section_image = self.get_free_image(result['title'])
                if section_image:
                    article_parts.append(f'<img src="{section_image}" alt="{result["title"]}" class="section-image">')
                    image_urls.append(section_image)
            
            return "\n".join(article_parts), image_urls
            
        except Exception as e:
            logger.error(f"Error generating article: {e}")
            return "", []

# Create a singleton instance
content_generator = ContentGenerator()
