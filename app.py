from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from components import ComponentManager, ComponentState
from bs4 import BeautifulSoup
import shutil
import requests
from typing import List, Dict, Optional
import time

# Configure logging
def setup_logging():
    """Configure logging to output to both file and console with the same format"""
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler('log.txt', mode='w', encoding='utf-8')
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # Ensure third-party loggers don't overwhelm our logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    return root_logger

logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(title="AgentZero", description="Self-improving web application")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create static files directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Initialize component manager
component_manager = ComponentManager()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"

async def call_llm_api(prompt: str) -> Optional[str]:
    """Call Ollama API with enhanced context"""
    try:
        # Log the prompt
        logger.info("Sending prompt to LLM")
        
        # Run the synchronous request in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(
            'http://localhost:11434/api/generate', 
            json={
                'model': 'qwen2.5-coder:7b',
                'prompt': prompt,
                'system': """You are an expert web developer assisting with web component improvements.
                IMPORTANT: You must ALWAYS return responses in this exact JSON format:
                {"html": "the improved HTML code here"}
                
                Guidelines:
                1. Return ONLY valid JSON, no other text or formatting
                2. The HTML must keep all data-component attributes
                3. Include CSS in <style> tags
                4. Use semantic HTML5 elements
                5. Focus on responsive design and accessibility
                
                Example response:
                {"html": "<div data-component='example'><style>...</style>...</div>"}""",
                'stream': False
            },
            timeout=30
        ))
        
        response.raise_for_status()
        result = response.json()['response']
        
        # Log the raw response for debugging
        logger.debug(f"Raw LLM response: {result}")
        
        logger.info("Received response from LLM")
        return result
        
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return None  # Return None instead of empty string

class AgentLoop:
    def __init__(self, component_manager: ComponentManager):
        self.component_manager = component_manager
        self.active_connections: list[WebSocket] = []
        self.is_running = False
        self.current_task = None
        self.min_improvement_interval = 5  # Minimum seconds between improvements
        self.last_improvement_time = None

    async def register(self, websocket: WebSocket):
        """Register a new WebSocket connection"""
        try:
            self.active_connections.append(websocket)
        except Exception as e:
            logger.error(f"Error registering websocket: {e}")

    async def unregister(self, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        except Exception as e:
            logger.error(f"Error unregistering websocket: {e}")

    async def broadcast_update(self, component_name: str, content: str, version: int):
        """Broadcast component updates to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    'type': 'component_update',
                    'component': component_name,
                    'content': content,
                    'version': version
                })
            except Exception as e:
                logger.error(f"Error broadcasting update: {e}")
                disconnected.append(connection)
                
        # Clean up disconnected clients
        for connection in disconnected:
            await self.unregister(connection)

    async def improvement_loop(self):
        """Main improvement loop"""
        try:
            while self.is_running:
                logger.info("Checking for improvements...")
                
                # Get current metrics
                metrics = self.component_manager.get_all_metrics()
                logger.info(f"Current metrics: \n{json.dumps(metrics, indent=2)}")
                
                # Get improvement suggestions
                suggestions = await get_improvement_suggestions(metrics)
                
                if not suggestions:
                    logger.info("No components need improvement")
                    if not self.is_running:
                        break
                    await asyncio.sleep(10)  # Wait before next check
                    continue
                
                # Process each suggestion
                for suggestion in suggestions:
                    if not self.is_running:
                        break
                        
                    component_name = suggestion['component']
                    score = suggestion['score']
                    areas = suggestion['areas']
                    priority = suggestion['priority']
                    
                    logger.info(f"Improving {component_name} (score: {score:.2f}, priority: {priority})")
                    logger.info(f"Areas needing improvement: {', '.join(areas)}")
                    
                    # Run improvement in a background task to avoid blocking
                    improvement_task = asyncio.create_task(improve_component(component_name, areas))
                    try:
                        await asyncio.wait_for(improvement_task, timeout=60)  # 60 second timeout
                    except asyncio.TimeoutError:
                        logger.error(f"Improvement task for {component_name} timed out")
                        continue
                    except Exception as e:
                        logger.error(f"Error in improvement task: {e}")
                        continue
                    
                    if not self.is_running:
                        break
                        
                    # Wait between improvements
                    await asyncio.sleep(self.min_improvement_interval)
                    
                if not self.is_running:
                    break
                    
                # Wait before next improvement cycle
                await asyncio.sleep(5)
                
            logger.info("Improvement loop stopped")
        except Exception as e:
            logger.error(f"Error in improvement loop: {e}")
            self.is_running = False
        finally:
            self.is_running = False
            logger.info("Improvement loop cleanup complete")

    async def start(self):
        """Start the improvement loop if not already running"""
        if not self.is_running:
            self.is_running = True
            self.current_task = asyncio.create_task(self.improvement_loop())
            return {'status': 'started'}

    async def stop(self):
        """Stop the improvement loop"""
        logger.info("Stopping improvement loop...")
        self.is_running = False
        
        # Cancel the current task if it exists
        if self.current_task and not self.current_task.done():
            try:
                self.current_task.cancel()
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                logger.error(f"Error cancelling current task: {e}")
        
        logger.info("Improvement loop stop requested")
        return {'status': 'stopped'}

# Initialize agent loop
agent = AgentLoop(component_manager)

@app.post("/start")
async def start_agent():
    """Start the agent improvement loop"""
    logger.info("Starting agent...")
    return await agent.start()

@app.post("/stop")
async def stop_agent():
    """Stop the agent improvement loop"""
    logger.info("Stopping agent...")
    return await agent.stop()

@app.post("/reset")
async def reset_state():
    """Reset the application state"""
    try:
        logger.info("Resetting state...")
        await agent.stop()  # Stop agent if running
        
        # Reset preview.html to index.html
        shutil.copy('templates/index.html', 'templates/preview.html')
        
        # Clear component history
        component_manager.reset()
        
        # Re-initialize components from preview template
        preview_html = get_current_preview()  # Now synchronous
        component_manager.extract_components(preview_html)
        
        # Evaluate all components
        for name in component_manager.components:
            component_manager.evaluate_component(name)
            logger.info(f"Initialized component {name}")
        
        logger.info("State reset complete")
        return {"status": "reset"}
    except Exception as e:
        logger.error(f"Error resetting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    # Create necessary directories
    for dir_name in ["static", "state", "components"]:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Create or clear log file
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('log.txt', 'w', encoding='utf-8') as f:
        f.write(f"=== New Session Started at {timestamp} ===\n")
    logger.info("Created/cleared log file")
    
    # Load initial components from preview template
    preview_path = Path("templates/preview.html")
    if not preview_path.exists():
        preview_path.write_text(DEFAULT_PREVIEW_TEMPLATE)
        
    # Initialize components from preview template
    preview_html = get_current_preview()  # Now synchronous
    component_manager.extract_components(preview_html)
    
    # Ensure all components are initialized
    for name in component_manager.components:
        logger.info(f"Initialized component {name}")
    
    logger.info("Application initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the improvement loop on shutdown"""
    logger.info("Starting shutdown sequence...")
    
    # Stop the agent and wait for it to complete
    try:
        await agent.stop()
        logger.info("Agent stopped")
    except Exception as e:
        logger.error(f"Error stopping agent: {e}")

    # Close all WebSocket connections
    try:
        for connection in manager.active_connections[:]:  # Create a copy of the list
            try:
                await connection.close()
                await manager.disconnect(connection)
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
        logger.info("All WebSocket connections closed")
    except Exception as e:
        logger.error(f"Error during WebSocket cleanup: {e}")

    # Cancel any pending tasks
    try:
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All pending tasks cancelled")
    except Exception as e:
        logger.error(f"Error cancelling pending tasks: {e}")

    logger.info("Shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/preview", response_class=HTMLResponse)
async def preview(request: Request):
    """Preview page showing live component updates and metrics"""
    # Ensure preview.html exists
    preview_path = Path("templates/preview.html")
    if not preview_path.exists():
        preview_path.write_text(DEFAULT_PREVIEW_TEMPLATE, encoding='utf-8')
        logger.info("Created preview template")
    
    # Get current metrics for all components
    preview_html = preview_path.read_text(encoding='utf-8')
    metrics = await evaluate_preview(preview_html)
    
    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "metrics": metrics
        }
    )

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Admin page with logs and status"""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    try:
        await websocket.accept()
        await agent.register(websocket)  # Register with AgentLoop
        
        # Re-read preview.html and extract components to ensure latest state
        preview_html = get_current_preview()
        component_manager.extract_components(preview_html)
        
        # Send current component states
        for name, component in component_manager.components.items():
            await websocket.send_json({
                'type': 'component_update',
                'component': name,
                'content': component.content,
                'version': component.version
            })
            logger.info(f"Sent initial state for component {name} version {component.version}")
        
        # Send initial log content
        try:
            with open('log.txt', 'r', encoding='utf-8', errors='replace') as f:
                log_content = f.read()
                await websocket.send_json({
                    'type': 'log_update',
                    'message': log_content
                })
        except Exception as e:
            logger.error(f"Error sending initial logs: {e}")
        
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get('type') == 'reset':
                    await reset_state()
                elif data.get('type') == 'start':
                    await start_agent()
                elif data.get('type') == 'stop':
                    await stop_agent()
                elif data.get('type') == 'refresh_preview':
                    # Allow client to request a fresh state
                    preview_html = get_current_preview()
                    component_manager.extract_components(preview_html)
                    # Send refreshed states
                    for name, component in component_manager.components.items():
                        await websocket.send_json({
                            'type': 'component_update',
                            'component': name,
                            'content': component.content,
                            'version': component.version
                        })
                elif data.get('type') == 'get_logs':
                    with open('log.txt', 'r', encoding='utf-8', errors='replace') as f:
                        new_content = f.read()
                        if new_content != getattr(websocket, '_last_log_content', None):
                            await websocket.send_json({
                                'type': 'log_update',
                                'message': new_content
                            })
                            setattr(websocket, '_last_log_content', new_content)
                            
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling websocket message: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await agent.unregister(websocket)  # Always unregister

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients"""
        for connection in self.active_connections:
            try:
                # Ensure message is JSON-serializable and handle encoding
                message_str = json.dumps(message, ensure_ascii=False)
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                continue

manager = ConnectionManager()

# Global variables
current_run_files = []

# Default template for the application
DEFAULT_PREVIEW_TEMPLATE = """{% extends "base.html" %}

{% block content %}
    <header data-component="header" class="container">
        <h1>Welcome to AgentZero</h1>
        <p>A self-improving web application</p>
    </header>

    <nav data-component="navigation" class="container">
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
        </ul>
    </nav>

    <section data-component="features" class="container">
        <h2>Key Features</h2>
        <ul>
            <li>Self-improving components</li>
            <li>Real-time updates</li>
            <li>Intelligent design decisions</li>
        </ul>
    </section>

    <main data-component="content" class="container">
        <h2>Main Content</h2>
        <p>This is the main content area that will be improved over time.</p>
    </main>

    <footer data-component="footer" class="container">
        <p>Copyright 2023 AgentZero. All rights reserved.</p>
    </footer>
{% endblock %}"""

def get_current_preview() -> str:
    """Get current contents of preview.html"""
    try:
        # First try to read preview.html
        preview_path = Path("templates/preview.html")
        if not preview_path.exists():
            # If preview doesn't exist, copy from index.html
            shutil.copy('templates/index.html', 'templates/preview.html')
        
        # Render the template to expand any Jinja2 blocks
        template = templates.get_template("preview.html")
        rendered_html = template.render()
        return rendered_html
        
    except Exception as e:
        logger.error(f"Error reading preview.html: {e}")
        # Render the default template as fallback
        template = templates.get_template("index.html")
        return template.render()

async def log_section(title: str, content: str):
    """Log a section with title and content"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        section = f"""
==================== {title} ====================
{content}
==================================================
"""
        # Write to log file with proper encoding
        with open('log.txt', 'a', encoding='utf-8', errors='replace') as f:
            f.write(f"\n{timestamp} - {section}")
            
        # Broadcast log update
        try:
            with open('log.txt', 'r', encoding='utf-8', errors='replace') as f:
                log_content = f.read()
                await manager.broadcast({
                    'type': 'log_update',
                    'message': log_content
                })
        except Exception as e:
            logger.error(f"Error broadcasting log update: {e}")
            
    except Exception as e:
        logger.error(f"Error writing to log file: {e}")

async def evaluate_preview(html_content: str) -> dict:
    """Evaluate components in the preview"""
    soup = BeautifulSoup(html_content, 'html.parser')
    metrics = {}
    
    for component in soup.find_all(attrs={'data-component': True}):
        component_name = component.get('data-component')
        if not component_name:
            continue
            
        metrics[component_name] = {
            'design': {
                'layout': evaluate_layout(component),
                'style': evaluate_style(component),
                'responsiveness': evaluate_responsiveness(component)
            },
            'content': {
                'text_quality': evaluate_text_content(component),
                'content_depth': evaluate_content_depth(component),
                'media_richness': evaluate_media_content(component)
            }
        }
    
    return metrics

def evaluate_text_content(component) -> float:
    """Evaluate the quality and completeness of text content"""
    text = component.get_text(strip=True)
    score = 1.0
    
    # Check for empty or very short content
    if not text:
        return 0.1
    if len(text) < 50:
        score *= 0.5
        
    # Check for placeholder text
    placeholder_patterns = ['lorem ipsum', 'placeholder', 'coming soon', 'tbd']
    if any(pattern in text.lower() for pattern in placeholder_patterns):
        score *= 0.3
        
    # Check for content structure
    has_headings = bool(component.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
    has_paragraphs = bool(component.find_all('p'))
    if has_headings and has_paragraphs:
        score *= 1.2
    
    return min(1.0, score)

def evaluate_content_depth(component) -> float:
    """Evaluate the depth and richness of content"""
    score = 1.0
    
    # Check for content hierarchy
    heading_levels = len(set(tag.name for tag in component.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])))
    if heading_levels > 1:
        score *= 1.2
    
    # Check for structured content
    has_lists = bool(component.find_all(['ul', 'ol']))
    has_tables = bool(component.find_all('table'))
    if has_lists or has_tables:
        score *= 1.1
        
    # Check for interactive elements
    has_buttons = bool(component.find_all('button'))
    has_forms = bool(component.find_all('form'))
    if has_buttons or has_forms:
        score *= 1.1
    
    return min(1.0, score)

def evaluate_media_content(component) -> float:
    """Evaluate the presence and quality of media content"""
    score = 1.0
    
    # Check for images
    images = component.find_all('img')
    if not images:
        score *= 0.6
    else:
        # Check for alt text
        images_with_alt = [img for img in images if img.get('alt')]
        if images_with_alt:
            score *= 1.1
            
    # Check for other media
    has_video = bool(component.find_all(['video', 'iframe']))
    has_audio = bool(component.find_all('audio'))
    if has_video or has_audio:
        score *= 1.2
        
    return min(1.0, score)

def evaluate_layout(component) -> float:
    """Evaluate the layout quality of a component"""
    score = 1.0
    
    # Check for semantic HTML structure
    semantic_tags = ['header', 'footer', 'nav', 'main', 'article', 'section', 'aside']
    if any(component.find_all(tag) for tag in semantic_tags):
        score *= 1.2
    
    # Check for grid/flex layout
    style = component.get('style', '').lower()
    classes = ' '.join(component.get('class', [])).lower()
    
    if any(term in style or term in classes for term in ['grid', 'flex']):
        score *= 1.1
    
    # Check for responsive units
    responsive_units = ['%', 'rem', 'em', 'vh', 'vw']
    if any(unit in style for unit in responsive_units):
        score *= 1.1
    else:
        score *= 0.8  # Penalize for not using responsive units
    
    return min(1.0, score)

def evaluate_style(component) -> float:
    """Evaluate the styling quality of a component"""
    score = 1.0
    
    style = component.get('style', '').lower()
    classes = ' '.join(component.get('class', [])).lower()
    
    # Check for custom styling
    if not style and not classes:
        score *= 0.7  # Penalize for no styling
    
    # Check for color usage
    color_props = ['color', 'background', 'border']
    if any(prop in style for prop in color_props):
        score *= 1.1
    
    # Check for typography
    typography_props = ['font-', 'text-', 'line-height']
    if any(prop in style for prop in typography_props):
        score *= 1.1
    
    # Check for transitions/animations
    animation_props = ['transition', 'animation', 'transform']
    if any(prop in style for prop in animation_props):
        score *= 1.1
    
    return min(1.0, score)

def evaluate_responsiveness(component) -> float:
    """Evaluate the responsive design of a component"""
    score = 1.0
    
    style = component.get('style', '').lower()
    classes = ' '.join(component.get('class', [])).lower()
    
    # Check for media queries
    if '@media' in style:
        score *= 1.2
    
    # Check for responsive units
    responsive_units = ['%', 'rem', 'em', 'vh', 'vw']
    if any(unit in style for unit in responsive_units):
        score *= 1.1
    
    # Check for responsive classes
    responsive_terms = ['sm:', 'md:', 'lg:', 'xl:', 'mobile', 'tablet', 'desktop']
    if any(term in classes for term in responsive_terms):
        score *= 1.1
    
    # Check for flex/grid
    if any(term in style or term in classes for term in ['flex', 'grid']):
        score *= 1.1
    
    return min(1.0, score)

async def get_improvement_suggestions(metrics: dict) -> List[dict]:
    """Get improvement suggestions for components based on metrics"""
    suggestions = []
    IMPROVEMENT_THRESHOLD = 0.85  # Increased from default 0.7
    
    for component_name, component_metrics in metrics.items():
        design_metrics = component_metrics.get('design', {})
        content_metrics = component_metrics.get('content', {})
        
        # Calculate average scores
        design_score = sum(design_metrics.values()) / len(design_metrics) if design_metrics else 0
        content_score = sum(content_metrics.values()) / len(content_metrics) if content_metrics else 0
        overall_score = (design_score + content_score) / 2
        
        logger.info(f"Component {component_name} score: {overall_score}")
        
        if overall_score < IMPROVEMENT_THRESHOLD:
            # Identify specific areas needing improvement
            improvements = []
            
            # Check design metrics
            if design_metrics.get('layout', 1.0) < 0.9:
                improvements.append("layout structure")
            if design_metrics.get('style', 1.0) < 0.9:
                improvements.append("visual styling")
            if design_metrics.get('responsiveness', 1.0) < 0.9:
                improvements.append("responsive design")
            
            # Check content metrics
            if content_metrics.get('text_quality', 1.0) < 0.9:
                improvements.append("text content quality")
            if content_metrics.get('content_depth', 1.0) < 0.9:
                improvements.append("content depth")
            if content_metrics.get('media_richness', 1.0) < 0.9:
                improvements.append("media integration")
            
            if improvements:
                suggestions.append({
                    'component': component_name,
                    'score': overall_score,
                    'areas': improvements,
                    'priority': 'high' if overall_score < 0.8 else 'medium'
                })
    
    # Sort suggestions by priority (high first) and then by score (lowest first)
    suggestions.sort(key=lambda x: (0 if x['priority'] == 'high' else 1, x['score']))
    return suggestions

async def improve_component(name: str, suggestions: List[str]):
    """Improve a specific component"""
    try:
        logger.info(f"Improving component {name} with suggestions: {suggestions}")
        
        # Get current component content
        component = component_manager.get_component(name)
        if not component:
            logger.error(f"Component {name} not found")
            return False
            
        # Prepare prompt for LLM
        prompt = f"""You are an expert web developer. Improve this HTML component following these requirements EXACTLY:

Current HTML:
{component.content}

Required Improvements:
{json.dumps(suggestions, indent=2)}

REQUIREMENTS:
1. Return ONLY a JSON object with this EXACT structure:
{{
    "html": "the improved HTML code here"
}}
2. The HTML must:
   - Keep the same data-component attribute
   - Implement ALL suggested improvements
   - Use semantic HTML5 elements
   - Include CSS in a <style> tag if needed
3. DO NOT include any explanation, markdown, or extra text
4. The response must be valid JSON
5. DO NOT modify or remove the data-component attribute

Example response:
{{"html": "<div data-component='example'><style>...</style>...</div>"}}"""

        logger.info("Sending prompt to LLM")
        raw_response = await call_llm_api(prompt)
        if not raw_response:
            logger.error("No response from LLM")
            return False
            
        logger.info(f"Raw LLM response: {json.dumps(raw_response, indent=4)}")
        
        # Clean and parse the response
        try:
            # Remove any markdown code blocks
            clean_response = raw_response.strip()
            if clean_response.startswith('```'):
                clean_response = ''.join(clean_response.split('```')[1:])
            if clean_response.startswith('json'):
                clean_response = clean_response[4:]
            clean_response = clean_response.strip()
            
            # Parse JSON
            response = json.loads(clean_response)
            
            if 'html' not in response:
                logger.error("No HTML in LLM response")
                return False
                
            # Update component with new content
            new_content = response['html']
            updated_component = component_manager.update_component(name, new_content)
            if not updated_component:
                logger.error(f"Failed to update component {name}")
                return False
            logger.info(f"Updated component {name} to version {updated_component.version}")
            
            # Broadcast preview update
            await agent.broadcast_update(name, new_content, updated_component.version)
            logger.info("Broadcast preview update")
            
            # Update preview.html
            preview_html = get_current_preview()
            soup = BeautifulSoup(preview_html, 'html.parser')
            old_component = soup.find(attrs={'data-component': name})
            if old_component:
                new_soup = BeautifulSoup(new_content, 'html.parser')
                old_component.replace_with(new_soup)
                preview_path = Path("templates/preview.html")
                preview_path.write_text(str(soup), encoding='utf-8')
                logger.info("Wrote changes to templates/preview.html")
                
                # Broadcast file history update
                await agent.broadcast_update('file_history', {
                    'file': 'preview.html',
                    'action': 'update',
                    'component': name
                }, 0)
                logger.info("Broadcast file history update")
                
            logger.info(f"Successfully improved component {name}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error improving component {name}: {e}")
        return False

async def calculate_component_score(metrics: Dict) -> float:
    """Calculate overall score for a component"""
    total_score = 0
    total_metrics = 0
    
    for category in metrics:
        for metric, score in metrics[category].items():
            total_score += score
            total_metrics += 1
            
    return total_score / total_metrics if total_metrics > 0 else 0.0

async def write_changes(changes: dict):
    """Write changes to files"""
    if not changes:
        await log_section("WARNING", "No changes to write")
        return
    
    try:
        for file_path, content in changes.items():
            await log_section(f"WRITING FILE: {file_path}", content)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # If this is preview.html, broadcast update first
            if file_path == 'templates/preview.html':
                try:
                    await manager.broadcast({
                        'type': 'preview_update',
                        'content': content
                    })
                    logger.info("Broadcast preview update")
                except Exception as e:
                    logger.error(f"Error broadcasting preview update: {e}")
            
            # Write file with proper encoding
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Wrote changes to {file_path}")
            except Exception as e:
                logger.error(f"Error writing to {file_path}: {e}")
                continue
            
            # Add to current run's files
            try:
                current_run_files.append({
                    'name': file_path,
                    'content': content,
                    'timestamp': int(datetime.now().timestamp()),
                    'type': 'current'
                })
                # Broadcast file history update
                await manager.broadcast({
                    'type': 'file_history',
                    'files': current_run_files
                })
                logger.info("Broadcast file history update")
            except Exception as e:
                logger.error(f"Error updating file history: {e}")
            
            await log_section("SUCCESS", f"Written to {file_path}")
            
    except Exception as e:
        await log_section("ERROR", f"Failed to write changes: {str(e)}")
        logger.error(f"Error in write_changes: {e}")

async def reset_workspace():
    """Reset workspace by archiving current files and creating fresh directories"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Archive current preview.html if it exists
        preview_path = os.path.join('templates', 'preview.html')
        if os.path.exists(preview_path):
            archive_dir = os.path.join('archive', f'preview_{timestamp}')
            os.makedirs(archive_dir, exist_ok=True)
            shutil.copy2(preview_path, os.path.join(archive_dir, 'preview.html'))
        
        # Create fresh preview.html with proper encoding
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write('''{% extends "base.html" %}

{% block content %}
    <header class="container" data-component="header">
        <h1>Welcome to AgentZero</h1>
        <p>A self-improving web application</p>
    </header>
    
    <nav class="container" data-component="navigation">
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
        </ul>
    </nav>
    
    <main class="container" data-component="content">
        <h2>Main Content</h2>
        <p>This content will be improved over time.</p>
    </main>
    
    <footer class="container" data-component="footer">
        <p>&copy; 2023 AgentZero. All rights reserved.</p>
    </footer>
{% endblock %}''')
        
        # Archive and clear directories
        for dir_name in ['once', 'sequence', 'library']:
            if os.path.exists(dir_name):
                archive_path = os.path.join('archive', f'{dir_name}_{timestamp}')
                if os.listdir(dir_name):  # Only archive if directory is not empty
                    shutil.copytree(dir_name, archive_path)
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
        
        # Clear log file with proper encoding
        with open('log.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== New Session Started at {timestamp} ===\n")
        
        # Clear current run's file history
        global current_run_files
        current_run_files = []
        
        logger.info("Workspace reset complete")
        
    except Exception as e:
        logger.error(f"Error resetting workspace: {e}")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import shutil
    
    # Run the app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
