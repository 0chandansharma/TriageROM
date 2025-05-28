#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class JSONHandler:
    """Handle JSON file operations for session data"""
    
    def __init__(self):
        pass
    
    def write_json_file(self, file_path: str, data: Dict, pretty_format: bool = True) -> bool:
        """
        Write data to JSON file
        
        Args:
            file_path: Output file path
            data: Data to write
            pretty_format: Whether to format JSON nicely
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata
            output_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "file_version": "1.0",
                    "generated_by": "TriageROM API"
                },
                "data": data
            }
            
            with open(file_path, 'w') as f:
                if pretty_format:
                    json.dump(output_data, f, indent=2, default=self._json_serializer)
                else:
                    json.dump(output_data, f, default=self._json_serializer)
            
            return True
            
        except Exception as e:
            print(f"Error writing JSON file: {e}")
            return False
    
    def read_json_file(self, file_path: str) -> Optional[Dict]:
        """
        Read JSON file
        
        Args:
            file_path: JSON file path
            
        Returns:
            Data dictionary or None if failed
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None
    
    def _json_serializer(self, obj: Any) -> str:
        """Custom JSON serializer for non-standard types"""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)