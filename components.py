from typing import List, Dict, Optional
import json
import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path
import logging

# Import evaluation functions
from metrics import (
    evaluate_layout,
    evaluate_style,
    evaluate_responsiveness,
    evaluate_text_content,
    evaluate_content_depth,
    evaluate_media_content
)

logger = logging.getLogger(__name__)

class ComponentState:
    def __init__(self, name: str, content: str, version: int = 1, metrics: Optional[Dict] = None):
        self.name = name
        self.content = content
        self.version = version
        self.metrics = metrics or {}
        self.history = []
        self.last_updated = datetime.now().isoformat()
    
    def update(self, new_content: str, metrics: Optional[Dict] = None):
        """Update component with new content and optionally new metrics"""
        self.history.append({
            'version': self.version,
            'content': self.content,
            'metrics': self.metrics,
            'timestamp': self.last_updated
        })
        self.version += 1
        self.content = new_content
        if metrics:
            self.metrics = metrics
        self.last_updated = datetime.now().isoformat()
    
    def rollback(self) -> bool:
        """Rollback to previous version"""
        if not self.history:
            return False
        
        last_state = self.history.pop()
        self.version = last_state['version']
        self.content = last_state['content']
        self.metrics = last_state['metrics']
        self.last_updated = last_state['timestamp']
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'content': self.content,
            'version': self.version,
            'metrics': self.metrics,
            'history': self.history,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ComponentState':
        """Create instance from dictionary"""
        instance = cls(data['name'], data['content'], data['version'], data['metrics'])
        instance.history = data['history']
        instance.last_updated = data['last_updated']
        return instance

class ComponentManager:
    def __init__(self, storage_dir: str = 'components'):
        self.storage_dir = storage_dir
        self.components: Dict[str, ComponentState] = {}
        self.component_history: Dict[str, List[ComponentState]] = {}  # Initialize history dict
        self._ensure_storage()
        self.load_components()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def _get_component_path(self, name: str) -> str:
        """Get path for component storage file"""
        return os.path.join(self.storage_dir, f"{name}.json")
    
    def load_components(self):
        """Load all components from storage"""
        if not os.path.exists(self.storage_dir):
            return
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                name = filename[:-5]  # Remove .json
                path = os.path.join(self.storage_dir, filename)
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.components[name] = ComponentState.from_dict(data)
                    self.component_history[name] = []
    
    def _save_component(self, component: ComponentState):
        """Save component to disk"""
        path = self._get_component_path(component.name)
        with open(path, 'w') as f:
            json.dump(component.to_dict(), f, indent=2)
        logger.info(f"Saved component {component.name} version {component.version}")
    
    def save_component(self, component: ComponentState):
        """Save component to storage"""
        self._save_component(component)
            
    def reset(self):
        """Reset the component manager state"""
        try:
            # Clear in-memory components
            self.components.clear()
            self.component_history.clear()
            
            # Clear component storage files
            for file in Path(self.storage_dir).glob('*.json'):
                file.unlink()
                
            logger.info("Component manager state reset")
            
        except Exception as e:
            logger.error(f"Error resetting component manager: {e}")
            raise
        
    def get_component(self, name: str) -> Optional[ComponentState]:
        """Get component by name"""
        return self.components.get(name)
    
    def update_component(self, name: str, content: str, metrics: Optional[Dict] = None) -> Optional[ComponentState]:
        """Update a component with new content and optionally new metrics"""
        component = self.get_component(name)
        if not component:
            return None
            
        # Store current state in history
        component.update(content, metrics)
        
        # Save to disk
        self._save_component(component)
        
        return component
    
    def add_component(self, name: str, content: str, metrics: Optional[Dict] = None) -> ComponentState:
        """Add a new component to the manager"""
        # Create new component
        component = ComponentState(name, content, version=1, metrics=metrics)
        
        # Add to components dict
        self.components[name] = component
        
        # Save to disk
        self._save_component(component)
        
        logger.info(f"Added component {name} version {component.version}")
        return component
        
    def get_or_create_component(self, name: str, content: str, metrics: Optional[Dict] = None) -> ComponentState:
        """Get existing component or create new one"""
        component = self.get_component(name)
        if component is None:
            component = self.add_component(name, content, metrics)
        return component
    
    def rollback_component(self, name: str) -> bool:
        """Rollback specific component to previous version"""
        if name not in self.components:
            return False
        
        success = self.components[name].rollback()
        if success:
            self._save_component(self.components[name])
        return success
    
    def extract_components(self, html: str) -> Dict[str, ComponentState]:
        """Extract components from HTML and update/create them"""
        soup = BeautifulSoup(html, 'html.parser')
        components = {}
        
        try:
            # Find all elements with data-component attribute
            component_elements = soup.find_all(attrs={'data-component': True})
            logger.info(f"Found {len(component_elements)} components in HTML")
            
            for element in component_elements:
                name = element['data-component']
                content = str(element)
                
                # Create or update component
                component = self.get_or_create_component(name, content)
                components[name] = component
                
            return components
            
        except Exception as e:
            logger.error(f"Error extracting components: {str(e)}")
            logger.error(f"HTML sample: {str(soup)[:200]}")
            return {}
    
    def compose_html(self, template_html: str) -> str:
        """Compose full HTML from template and components"""
        soup = BeautifulSoup(template_html, 'html.parser')
        
        # Replace each component placeholder with actual content
        for element in soup.find_all(attrs={'data-component': True}):
            name = element['data-component']
            if name in self.components:
                # Parse component content to ensure proper integration
                component_soup = BeautifulSoup(self.components[name].content, 'html.parser')
                element.replace_with(component_soup)
        
        return str(soup)

    def get_lowest_scoring_component(self) -> Optional[str]:
        """Get the name of the component with the lowest overall score"""
        scores = {}
        for name in self.components:
            metrics = self.evaluate_component(name)
            if metrics:
                # Calculate average score across all metrics
                score = sum(metrics.values()) / len(metrics)
                scores[name] = score
                logger.info(f"Component {name} score: {score:.2f}")
        
        if not scores:
            logger.warning("No components found to evaluate")
            return None
            
        # Get component with lowest score
        lowest_component = min(scores.items(), key=lambda x: x[1])
        logger.info(f"Lowest scoring component: {lowest_component[0]} ({lowest_component[1]:.2f})")
        return lowest_component[0]

    def generate_improvements(self, component_name: str) -> List[str]:
        """Generate improvement suggestions for a component based on its metrics"""
        if component_name not in self.components:
            return []

        current_metrics = self.evaluate_component(component_name)
        if not current_metrics:
            return []

        suggestions = []
        component = self.components[component_name]
        soup = BeautifulSoup(component.content, 'html.parser')

        # Color contrast improvements
        if current_metrics.get('color_contrast', 1.0) < 0.7:
            suggestions.append("Improve color contrast by using more distinct colors")
            suggestions.append("Consider using a complementary color scheme")

        # Typography improvements
        if current_metrics.get('typography', 1.0) < 0.7:
            if len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])) < 2:
                suggestions.append("Add proper heading hierarchy")
            if not any('font-family' in str(tag) for tag in soup.find_all('style')):
                suggestions.append("Define consistent font families")

        # Spacing improvements
        if current_metrics.get('spacing', 1.0) < 0.7:
            if not any('margin' in str(tag) or 'padding' in str(tag) for tag in soup.find_all(True)):
                suggestions.append("Add proper spacing between elements")
            suggestions.append("Use consistent spacing units (rem/em)")

        # Responsiveness improvements
        if current_metrics.get('responsiveness', 1.0) < 0.7:
            if not any('display: flex' in str(tag) or 'display: grid' in str(tag) for tag in soup.find_all('style')):
                suggestions.append("Implement flexible layouts using CSS Grid or Flexbox")
            if not any('@media' in str(tag) for tag in soup.find_all('style')):
                suggestions.append("Add media queries for responsive design")

        # Interactivity improvements
        if current_metrics.get('interactivity', 1.0) < 0.7:
            if len(soup.find_all(['button', 'a', 'input'])) < 3:
                suggestions.append("Add more interactive elements")
            if not any(':hover' in str(tag) or ':focus' in str(tag) for tag in soup.find_all('style')):
                suggestions.append("Add hover and focus states")

        # Content structure improvements
        if current_metrics.get('content_structure', 1.0) < 0.7:
            semantic_tags = ['header', 'nav', 'main', 'article', 'section', 'aside', 'footer']
            if not any(soup.find(tag) for tag in semantic_tags):
                suggestions.append("Use semantic HTML elements")
            if len(soup.find_all('p')) < 2:
                suggestions.append("Add more structured content")

        return suggestions

    def apply_improvements(self, component_name: str, suggestions: List[str]) -> Optional[str]:
        """Apply improvement suggestions to a component"""
        if component_name not in self.components:
            return None

        component = self.components[component_name]
        soup = BeautifulSoup(component.content, 'html.parser')

        # Apply improvements based on suggestions
        for suggestion in suggestions:
            if "color contrast" in suggestion.lower():
                # Implement color contrast improvements
                style = soup.find('style')
                if not style:
                    style = soup.new_tag('style')
                    soup.head.append(style) if soup.head else soup.append(style)
                style.string = """
                    :root {
                        --primary: #2563eb;
                        --secondary: #0ea5e9;
                        --accent: #6366f1;
                        --background: #f8fafc;
                        --text: #1e293b;
                    }
                """

            elif "heading hierarchy" in suggestion.lower():
                # Implement heading hierarchy
                if not soup.find('h1'):
                    title = soup.find(class_='title') or soup.find('p')
                    if title:
                        new_h1 = soup.new_tag('h1')
                        new_h1.string = title.string
                        title.replace_with(new_h1)

            elif "spacing" in suggestion.lower():
                # Implement consistent spacing
                style = soup.find('style')
                if not style:
                    style = soup.new_tag('style')
                    soup.head.append(style) if soup.head else soup.append(style)
                current_styles = style.string or ""
                style.string = current_styles + """
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    .container {
                        padding: 2rem;
                        margin: 0 auto;
                        max-width: 1200px;
                    }
                    .spacing {
                        margin-bottom: 1.5rem;
                    }
                """

            elif "flexible layouts" in suggestion.lower():
                # Implement responsive layouts
                style = soup.find('style')
                if not style:
                    style = soup.new_tag('style')
                    soup.head.append(style) if soup.head else soup.append(style)
                current_styles = style.string or ""
                style.string = current_styles + """
                    .flex-container {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 1rem;
                    }
                    .grid-container {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 1rem;
                    }
                    @media (max-width: 768px) {
                        .flex-container {
                            flex-direction: column;
                        }
                    }
                """

            elif "interactive elements" in suggestion.lower():
                # Add interactive elements
                style = soup.find('style')
                if not style:
                    style = soup.new_tag('style')
                    soup.head.append(style) if soup.head else soup.append(style)
                current_styles = style.string or ""
                style.string = current_styles + """
                    .button {
                        padding: 0.5rem 1rem;
                        border-radius: 0.25rem;
                        background: var(--primary);
                        color: white;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .button:hover {
                        background: var(--primary-dark);
                        transform: translateY(-1px);
                    }
                    .button:focus {
                        outline: 2px solid var(--primary);
                        outline-offset: 2px;
                    }
                """

            elif "semantic HTML" in suggestion.lower():
                # Add semantic structure
                main_content = soup.find(class_='content') or soup.find('div')
                if main_content:
                    article = soup.new_tag('article')
                    article['class'] = 'content-article'
                    article.extend(main_content.contents)
                    main_content.replace_with(article)

        return str(soup)

    def evaluate_component(self, component_name: str) -> Dict[str, float]:
        """Evaluate all metrics for a component"""
        if component_name not in self.components:
            logger.warning(f"Component {component_name} not found")
            return {}

        component = self.components[component_name]
        metrics = {}
        
        try:
            soup = BeautifulSoup(component.content, 'html.parser')
            
            # Design metrics
            design_metrics = {
                'layout': evaluate_layout(soup),
                'style': evaluate_style(soup),
                'responsiveness': evaluate_responsiveness(soup)
            }
            
            # Content metrics
            content_metrics = {
                'text_quality': evaluate_text_content(soup),
                'content_depth': evaluate_content_depth(soup),
                'media_richness': evaluate_media_content(soup)
            }
            
            metrics = {
                'design': design_metrics,
                'content': content_metrics
            }
            
            # Store metrics in component state
            component.metrics = metrics
            self._save_component(component)
            
            logger.info(f"Evaluated metrics for {component_name}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating metrics for {component_name}: {e}")
            return {}

    def get_all_metrics(self) -> dict:
        """Get metrics for all components"""
        all_metrics = {}
        for name, component in self.components.items():
            # If component doesn't have metrics, evaluate them
            if not component.metrics:
                self.evaluate_component(name)
            all_metrics[name] = component.metrics
            
        logger.info(f"All component metrics: {json.dumps(all_metrics, indent=2)}")
        return all_metrics

def evaluate_color_contrast(html: str) -> float:
    """Evaluate color contrast in a component"""
    soup = BeautifulSoup(html, 'html.parser')
    style_tags = soup.find_all('style')
    inline_styles = [el.get('style', '') for el in soup.find_all(attrs={'style': True})]
    
    # Extract colors from style tags and inline styles
    color_pattern = r'(?:color|background(?:-color)?)\s*:\s*(#[0-9a-fA-F]{6}|rgb\([^)]+\)|[a-zA-Z]+)'
    colors = []
    
    for style in style_tags:
        colors.extend(re.findall(color_pattern, style.string or ''))
    for style in inline_styles:
        colors.extend(re.findall(color_pattern, style))
    
    # Score based on number of unique colors (too few or too many is bad)
    unique_colors = len(set(colors))
    if unique_colors == 0:
        return 0.0
    elif unique_colors < 3:
        return 0.3
    elif unique_colors <= 5:
        return 1.0
    else:
        return max(0.0, 1.0 - (unique_colors - 5) * 0.1)

def evaluate_typography(html: str) -> float:
    """Evaluate typography usage"""
    soup = BeautifulSoup(html, 'html.parser')
    style_tags = soup.find_all('style')
    inline_styles = [el.get('style', '') for el in soup.find_all(attrs={'style': True})]
    
    # Check font families
    font_pattern = r'font-family\s*:\s*([^;]+)'
    fonts = []
    
    for style in style_tags:
        fonts.extend(re.findall(font_pattern, style.string or ''))
    for style in inline_styles:
        fonts.extend(re.findall(font_pattern, style))
    
    # Check heading hierarchy
    headings = [len(soup.find_all(f'h{i}')) for i in range(1, 7)]
    has_proper_hierarchy = all(h1 >= h2 for h1, h2 in zip(headings[:-1], headings[1:]))
    
    # Score based on font variety and heading hierarchy
    font_score = min(1.0, len(set(fonts)) / 3)  # Ideal: 2-3 font families
    hierarchy_score = 1.0 if has_proper_hierarchy else 0.5
    
    return (font_score + hierarchy_score) / 2

def evaluate_spacing(html: str) -> float:
    """Evaluate spacing and layout"""
    soup = BeautifulSoup(html, 'html.parser')
    style_tags = soup.find_all('style')
    inline_styles = [el.get('style', '') for el in soup.find_all(attrs={'style': True})]
    
    # Check margin and padding usage
    spacing_pattern = r'(?:margin|padding)(?:-(?:top|right|bottom|left))?\s*:\s*([^;]+)'
    spacing_values = []
    
    for style in style_tags:
        spacing_values.extend(re.findall(spacing_pattern, style.string or ''))
    for style in inline_styles:
        spacing_values.extend(re.findall(spacing_pattern, style))
    
    # Score based on consistent spacing units
    if not spacing_values:
        return 0.0
    
    units = set(re.findall(r'[a-z]+|%', ''.join(spacing_values)))
    consistent_units = len(units) <= 2  # Ideally using rem/em + %
    
    return 1.0 if consistent_units else 0.5

def evaluate_responsiveness(html):
    """Evaluate responsive design implementation"""
    # If html is a string, parse it, otherwise use as is
    soup = BeautifulSoup(html, 'html.parser') if isinstance(html, str) else html
    
    # Check for media queries in style tags and inline styles
    media_queries = []
    style_tags = soup.find_all('style')
    for style in style_tags:
        media_queries.extend(re.findall(r'@media[^{]+{([^}]+)}', style.string or ''))
    
    # Check inline styles for responsive features
    elements = soup.find_all(lambda tag: tag.get('style', ''))
    for elem in elements:
        style = elem.get('style', '')
        # Check for flex/grid
        if re.search(r'display\s*:\s*(flex|grid)', style):
            media_queries.append(style)  # Count as responsive feature
        # Check for relative units
        if re.search(r':\s*\d+(?:rem|em|vh|vw|%)', style):
            media_queries.append(style)  # Count as responsive feature
    
    # Check for responsive classes
    responsive_classes = []
    for tag in soup.find_all():
        classes = tag.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()
        for cls in classes:
            if any(term in cls.lower() for term in ['sm-', 'md-', 'lg-', 'xl-', 'mobile-', 'tablet-', 'desktop-', 'responsive']):
                responsive_classes.append(cls)
    
    # Check for responsive attributes
    responsive_attrs = []
    for tag in soup.find_all():
        if tag.get('srcset') or tag.get('sizes'):
            responsive_attrs.append(tag)
            
    # Score based on responsive features
    score = 0.0
    if media_queries: score += 0.4
    if responsive_classes: score += 0.3
    if responsive_attrs: score += 0.3
    
    # Give partial credit for having at least some responsive features
    if not score and (elements or responsive_classes or responsive_attrs):
        score = 0.2
        
    return score

def evaluate_interactivity(html: str) -> float:
    """Evaluate interactive elements"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Count interactive elements
    buttons = len(soup.find_all('button'))
    links = len(soup.find_all('a'))
    forms = len(soup.find_all('form'))
    inputs = len(soup.find_all('input'))
    
    # Check for hover/focus styles
    style_tags = soup.find_all('style')
    hover_focus_styles = []
    for style in style_tags:
        hover_focus_styles.extend(re.findall(r':(?:hover|focus|active)[^{]*{([^}]+)}', style.string or ''))
    
    # Score based on interactive elements and their styling
    has_interactive = buttons + links + forms + inputs > 0
    has_hover_styles = len(hover_focus_styles) > 0
    
    if not has_interactive:
        return 0.0
    
    base_score = min(1.0, (buttons + links + forms + inputs) / 5)  # Ideal: 5+ interactive elements
    style_bonus = 0.3 if has_hover_styles else 0.0
    
    return min(1.0, base_score + style_bonus)

def evaluate_content_structure(html: str) -> float:
    """Evaluate content structure and hierarchy"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Check semantic elements
    semantic_tags = ['header', 'nav', 'main', 'article', 'section', 'aside', 'footer']
    semantic_count = sum(len(soup.find_all(tag)) for tag in semantic_tags)
    
    # Check list usage
    lists = len(soup.find_all(['ul', 'ol']))
    
    # Check paragraph structure
    paragraphs = len(soup.find_all('p'))
    
    # Score based on semantic structure
    semantic_score = min(1.0, semantic_count / 3)  # Ideal: 3+ semantic elements
    list_score = min(1.0, lists / 2)  # Ideal: 2+ lists
    paragraph_score = min(1.0, paragraphs / 3)  # Ideal: 3+ paragraphs
    
    return (semantic_score * 0.5 + list_score * 0.25 + paragraph_score * 0.25)

# Update component metrics with implemented functions
COMPONENT_METRICS = {
    "header": {
        "visual_appeal": {
            "color_contrast": evaluate_color_contrast,
            "typography": evaluate_typography,
            "spacing": evaluate_spacing
        },
        "responsiveness": {
            "mobile_layout": evaluate_responsiveness,
            "interactive_elements": evaluate_interactivity
        }
    },
    "navigation": {
        "interactivity": {
            "user_engagement": evaluate_interactivity,
            "responsive_design": evaluate_responsiveness
        },
        "structure": {
            "semantic_structure": evaluate_content_structure,
            "spacing": evaluate_spacing
        }
    },
    "features": {
        "content_structure": {
            "semantic_structure": evaluate_content_structure,
            "typography": evaluate_typography
        },
        "interactivity": {
            "user_engagement": evaluate_interactivity,
            "responsive_design": evaluate_responsiveness
        }
    },
    "content": {
        "typography": {
            "text_styling": evaluate_typography,
            "spacing": evaluate_spacing
        },
        "structure": {
            "content_organization": evaluate_content_structure,
            "responsiveness": evaluate_responsiveness
        }
    },
    "footer": {
        "layout": {
            "spacing": evaluate_spacing,
            "responsiveness": evaluate_responsiveness
        },
        "content": {
            "structure": evaluate_content_structure,
            "typography": evaluate_typography
        }
    }
}
