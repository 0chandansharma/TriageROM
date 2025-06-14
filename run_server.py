#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct server runner without module imports
"""

import sys
import os
from pathlib import Path

# Set up the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Import and run
if __name__ == "__main__":
    from api.main import app
    import uvicorn
    
    print("Starting TriageROM API Server")
    print("============================")
    print(f"Working directory: {os.getcwd()}")
    print("Server will be available at:")
    print("  http://localhost:8000/docs")
    print("  http://localhost:8000/redoc")
    print()s
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )