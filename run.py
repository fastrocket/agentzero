import uvicorn
import logging
import shutil
from pathlib import Path
import argparse
import signal
import sys

def signal_handler(sig, frame):
    """Handle interrupt signal"""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def clean_state():
    """Clean all state directories"""
    dirs_to_clean = ["static", "state", "components"]
    for dir_name in dirs_to_clean:
        path = Path(dir_name)
        if path.exists():
            shutil.rmtree(path)
            logging.info(f"Cleaned {dir_name} directory")

def ensure_directories():
    """Ensure required directories exist"""
    required_dirs = ["static", "templates", "state", "components"]
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description='Run the AgentZero application')
    parser.add_argument('--clean', action='store_true', help='Clean state before starting')
    args = parser.parse_args()

    if args.clean:
        clean_state()
    
    ensure_directories()
    
    config = uvicorn.Config(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        loop="asyncio",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )
    server = uvicorn.Server(config)
    server.run()
