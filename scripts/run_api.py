#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the TriageROM API server
"""

import sys
import os
import uvicorn
from pathlib import Path
import argparse

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project directory
os.chdir(project_root)

from utils.config_loader import ConfigLoader

def main():
    """Run the TriageROM API server"""
    parser = argparse.ArgumentParser(description="Run TriageROM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", default="default_config", help="Configuration file name")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.merge_configs(args.config, "mediapipe_config", "loweback_config")
    
    # Override with command line arguments
    host = args.host or config.get("api", {}).get("host", "0.0.0.0")
    port = args.port or config.get("api", {}).get("port", 8000)
    reload = args.reload or config.get("api", {}).get("debug", False)
    log_level = args.log_level or config.get("api", {}).get("log_level", "info")
    
    print("Starting TriageROM API Server")
    print("=============================")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Log Level: {log_level}")
    print()
    print("API Documentation will be available at:")
    print(f"  http://{host}:{port}/docs")
    print(f"  http://{host}:{port}/redoc")
    print()
    print("WebSocket endpoint:")
    print(f"  ws://{host}:{port}/ws/live-analysis/{{session_id}}")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path includes: {project_root}")
    print()
    
    try:
        # Use the absolute module path
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level.lower(),
            reload_dirs=[str(project_root)] if reload else None
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main()