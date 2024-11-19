import os
import glob
import subprocess
import requests
import time
import logging
import re
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import asyncio
import shutil
import uvicorn
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")

manager = ConnectionManager()

# Global variables
current_run_files = []

async def log_section(title: str, content: str):
    """Log a section with title and content"""
    message = f"\n{'='*20} {title} {'='*20}\n{content}\n{'='*50}\n"
    logger.info(message)
    with open('log.txt', 'a') as log_file:
        log_file.write(message)
    # Send update via WebSocket
    await manager.broadcast({
        'type': 'log_update',
        'message': open('log.txt', 'r').read()
    })

def reset_workspace():
    """Reset workspace by archiving current files and creating fresh directories"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Archive current preview.html if it exists
    preview_path = os.path.join('templates', 'preview.html')
    if os.path.exists(preview_path):
        archive_dir = os.path.join('archive', f'preview_{timestamp}')
        os.makedirs(archive_dir, exist_ok=True)
        shutil.copy2(preview_path, os.path.join(archive_dir, 'preview.html'))
    
    # Create fresh preview.html
    with open(preview_path, 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>AI Tools Hub</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Welcome to AI Tools Hub</h1>
    <p>This page will be improved by the AI agent.</p>
</body>
</html>''')
    
    # Archive and clear directories
    for dir_name in ['once', 'sequence', 'library']:
        if os.path.exists(dir_name):
            archive_path = os.path.join('archive', f'{dir_name}_{timestamp}')
            if os.listdir(dir_name):  # Only archive if directory is not empty
                shutil.copytree(dir_name, archive_path)
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    
    # Clear log file
    with open('log.txt', 'w') as f:
        f.write(f"=== New Session Started at {timestamp} ===\n")
    
    # Clear current run's file history
    global current_run_files
    current_run_files = []

# Routes
@app.get("/", response_class=HTMLResponse)
async def admin(request: Request):
    """Admin page with logs and status"""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/preview", response_class=HTMLResponse)
async def preview(request: Request):
    """Preview page showing the current SPA"""
    return templates.TemplateResponse("preview.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial data
        with open('log.txt', 'r') as f:
            await websocket.send_json({
                'type': 'log_update',
                'message': f.read()
            })
        await websocket.send_json({
            'type': 'file_history',
            'files': current_run_files
        })
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            if data == "get_updates":
                with open('log.txt', 'r') as f:
                    await websocket.send_json({
                        'type': 'log_update',
                        'message': f.read()
                    })
                await websocket.send_json({
                    'type': 'file_history',
                    'files': current_run_files
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# Start up
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    # Ensure required directories exist
    for dir_name in ['archive', 'once', 'sequence', 'library', 'static']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Reset workspace
    reset_workspace()
    
    # Start the agent loop
    asyncio.create_task(agent_loop())

async def agent_loop():
    """Main agent loop"""
    while True:
        try:
            await log_section("NEW CYCLE", "Starting new improvement cycle")
            
            # Get current preview content
            preview_content = await get_current_preview()
            
            # Evaluate current state
            scores = await evaluate_preview(preview_content)
            suggestions = await get_improvement_suggestions(scores)
            
            # Add evaluation results to prompt
            metrics_summary = "\nCurrent Metrics:\n"
            for category, metrics in scores.items():
                metrics_summary += f"\n{category.title()}:\n"
                for name, score in metrics.items():
                    metrics_summary += f"- {name}: {score}\n"
            
            suggestions_text = "\nSuggested Improvements:\n" + "\n".join(f"- {s}" for s in suggestions)
            
            # Prepare context for LLM
            full_prompt = f"""Your task is to improve this AI Tools Hub website. Current content:

```html
{preview_content}
```

{metrics_summary}
{suggestions_text}

You can:
1. Directly modify the preview.html by providing HTML code between ```html blocks
2. Create Python scripts by providing code between ```python blocks

The preview.html file is located at 'templates/preview.html'.
Python scripts will be placed in the 'once/' directory.

Focus on making the website more:
1. Visually appealing with modern design (add CSS!)
2. User-friendly and responsive
3. Informative about AI tools
4. Well-structured and maintainable

Make ONE meaningful improvement based on the metrics and suggestions above.
If you add CSS, put it in a <style> tag in the HTML for now.
IMPORTANT: Always provide complete HTML including <html>, <head>, and <body> tags.
"""
            
            # Call Ollama model
            improvement_suggestions = await call_llm_api(full_prompt)

            if improvement_suggestions:
                changes = await parse_model_response(improvement_suggestions)
                if changes:
                    await write_changes(changes)
            
            await log_section("CYCLE COMPLETE", "Waiting 5 seconds before next run...")
            await asyncio.sleep(5)

        except Exception as e:
            await log_section("ERROR", f"Error in main loop: {str(e)}")
            await asyncio.sleep(5)

async def get_current_preview():
    """Get current contents of preview.html"""
    try:
        with open(os.path.join('templates', 'preview.html'), 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading preview.html: {str(e)}")
        return ""

async def call_llm_api(prompt: str):
    """Call Ollama API with enhanced context"""
    await log_section("PROMPT TO LLM", prompt)
    
    try:
        response = requests.post('http://localhost:11434/api/generate', 
            json={'model': 'qwen2.5-coder:7b', 'prompt': prompt},
            stream=True)
        
        # Collect all response chunks
        full_response = ""
        for line in response.iter_lines():
            if line:
                # Parse each line as a JSON object
                chunk = json.loads(line)
                if 'response' in chunk:
                    full_response += chunk['response']
                if 'error' in chunk:
                    raise Exception(chunk['error'])
        
        await log_section("LLM RESPONSE", full_response)
        return full_response
    except Exception as e:
        error_msg = f"Error calling LLM API: {str(e)}"
        await log_section("ERROR", error_msg)
        return None

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
                await manager.broadcast({
                    'type': 'preview_update',
                    'content': content
                })
            
            # Write file
            with open(file_path, 'w') as f:
                f.write(content)
            
            # Add to current run's files
            current_run_files.append({
                'name': file_path,
                'content': content,
                'timestamp': int(datetime.now().timestamp()),
                'type': 'current'
            })
            await log_section("SUCCESS", f"Written to {file_path}")
            
            # Broadcast file history update
            await manager.broadcast({
                'type': 'file_history',
                'files': current_run_files
            })
    except Exception as e:
        await log_section("ERROR", f"Failed to write changes: {str(e)}")

# Evaluation metrics
METRICS = {
    'visual_appeal': {
        'images': lambda html: len(re.findall(r'<img\s+[^>]*src=[^>]+>', html)),
        'colors': lambda html: len(re.findall(r'(?:color|background):\s*[^;]+;', html)),
        'fonts': lambda html: len(re.findall(r'font-family:\s*[^;]+;', html))
    },
    'content': {
        'headings': lambda html: len(re.findall(r'<h[1-6][^>]*>', html)),
        'paragraphs': lambda html: len(re.findall(r'<p[^>]*>', html)),
        'lists': lambda html: len(re.findall(r'<[ou]l[^>]*>', html))
    },
    'interactivity': {
        'buttons': lambda html: len(re.findall(r'<button[^>]*>', html)),
        'links': lambda html: len(re.findall(r'<a[^>]*>', html)),
        'forms': lambda html: len(re.findall(r'<form[^>]*>', html))
    },
    'responsiveness': {
        'media_queries': lambda html: len(re.findall(r'@media[^{]+{[^}]+}', html)),
        'flex_grid': lambda html: len(re.findall(r'(?:display:\s*(?:flex|grid)|flex:|grid:)[^;]+;', html))
    }
}

async def evaluate_preview(html: str) -> dict:
    """Evaluate current preview against metrics"""
    scores = {}
    for category, metrics in METRICS.items():
        scores[category] = {}
        for name, metric_fn in metrics.items():
            scores[category][name] = metric_fn(html)
    
    # Broadcast metrics update
    await manager.broadcast({
        'type': 'metrics_update',
        'metrics': scores,
        'suggestions': await get_improvement_suggestions(scores)
    })
    
    return scores

async def get_improvement_suggestions(scores: dict) -> list:
    """Generate improvement suggestions based on metrics"""
    suggestions = []
    
    # Check for missing or low scores
    if scores['visual_appeal']['images'] < 1:
        suggestions.append("Add relevant images from a free API like Unsplash")
    if scores['visual_appeal']['colors'] < 3:
        suggestions.append("Add more color variety using a cohesive color scheme")
    if scores['content']['headings'] < 3:
        suggestions.append("Add more section headings for better content structure")
    if scores['interactivity']['buttons'] + scores['interactivity']['links'] < 3:
        suggestions.append("Add interactive elements like buttons or links")
    if scores['responsiveness']['media_queries'] < 1:
        suggestions.append("Add responsive design with media queries")
    
    return suggestions

async def parse_model_response(response: str) -> dict:
    """Parse LLM response into file changes"""
    changes = {}
    
    # Look for HTML content between triple backticks
    html_pattern = r"```html\n(.*?)\n```"
    html_match = re.search(html_pattern, response, re.DOTALL)
    
    if html_match:
        html_content = html_match.group(1).strip()
        # Update preview.html
        changes['templates/preview.html'] = html_content
        await log_section("PARSED HTML", html_content)
    
    # Look for Python content between triple backticks
    python_pattern = r"```python\n(.*?)\n```"
    python_matches = re.finditer(python_pattern, response, re.DOTALL)
    
    for i, match in enumerate(python_matches):
        python_content = match.group(1).strip()
        # Put Python files in once directory
        changes[f'once/improvement_{i+1}.py'] = python_content
        await log_section(f"PARSED PYTHON {i+1}", python_content)
    
    return changes

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import shutil
    
    # Run the app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
