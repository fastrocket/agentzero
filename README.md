# Mistral MVP - Self-Improving AI Web Application

A self-improving single-page web application that uses AI to iteratively enhance its own design and functionality.

## Features

- Autonomous AI-driven web design improvements
- Real-time preview updates via WebSocket
- Metrics-based evaluation system
- Admin interface for monitoring improvements
- Comprehensive logging system

## Requirements

- Python 3.9+
- Ollama with qwen2.5-coder:7b model
- FastAPI
- WebSocket support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fastrocket/mistralmvp.git
cd mistralmvp
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Ollama server with qwen2.5-coder:7b model

5. Run the application:
```bash
python app.py
```

## Project Structure

- `/templates/`: HTML templates
  - `admin.html`: Admin interface
  - `preview.html`: Live SPA preview
- `/archive/`: Historical versions
- `/once/`: Staging for improvements
- `/sequence/`: Ordered improvement files
- `/library/`: Reusable components

## Usage

1. Access the admin interface at `http://localhost:8000/`
2. View the live preview at `http://localhost:8000/preview`
3. Monitor improvements in real-time through the admin interface

## License

MIT
