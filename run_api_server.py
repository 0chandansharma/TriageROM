# Create run_api_server.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

if __name__ == "__main__":
    import uvicorn
    
    print("Starting TriageROM API Server...")
    print("API will be available at: http://localhost:8000")
    print("Documentation at: http://localhost:8000/docs")
    
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to True for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")