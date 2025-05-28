#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import Dict, List, Optional
from pathlib import Path
import uuid

class SessionManager:
    """Manage analysis sessions and data persistence"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sessions = {}
        self.data_directory = Path(config.get('data_directory', './session_data'))
        self.data_directory.mkdir(exist_ok=True)
        
        # Session configuration
        self.max_session_duration = config.get('max_session_duration_minutes', 30) * 60  # Convert to seconds
        self.auto_cleanup = config.get('auto_cleanup_sessions', True)
        
    def create_session(self, session_type: str = "lumbar_rom") -> str:
        """
        Create a new analysis session
        
        Args:
            session_type: Type of analysis session
            
        Returns:
            Unique session ID
        """
        session_id = f"{session_type}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        session_data = {
            "session_id": session_id,
            "session_type": session_type,
            "created_at": time.time(),
            "last_activity": time.time(),
            "status": "active",
            "movement_data": [],
            "analysis_results": None,
            "file_paths": {
                "mot_file": None,
                "json_data": None
            }
        }
        
        self.sessions[session_id] = session_data
        
        # Create session directory
        session_dir = self.data_directory / session_id
        session_dir.mkdir(exist_ok=True)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data by ID"""
        if session_id in self.sessions:
            return self.sessions[session_id].copy()
        return None
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for session"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = time.time()
    
    def add_movement_data(self, session_id: str, movement_data: Dict):
        """Add movement data to session"""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id]["movement_data"].append({
            "timestamp": time.time(),
            "data": movement_data
        })
        
        self.update_session_activity(session_id)
        return True
    
    def complete_session(self, session_id: str, analysis_results: Dict) -> Dict:
        """
        Complete a session and generate final files
        
        Args:
            session_id: Session identifier
            analysis_results: Final analysis results
            
        Returns:
            File generation results
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        session["status"] = "completed"
        session["analysis_results"] = analysis_results
        session["completed_at"] = time.time()
        
        # Generate files
        file_results = self._generate_session_files(session_id)
        
        return file_results
    
    def _generate_session_files(self, session_id: str) -> Dict:
        """Generate MOT and JSON files for completed session"""
        
        session = self.sessions[session_id]
        session_dir = self.data_directory / session_id
        
        file_results = {
            "files_generated": [],
            "errors": []
        }
        
        try:
            # Generate MOT file
            if self.config.get('auto_save_mot', True):
                mot_result = self._generate_mot_file(session, session_dir)
                if mot_result["success"]:
                    file_results["files_generated"].append(mot_result["file_path"])
                    session["file_paths"]["mot_file"] = str(mot_result["file_path"])
                else:
                    file_results["errors"].append(f"MOT file generation failed: {mot_result['error']}")
            
            # Generate JSON data file
            json_result = self._generate_json_file(session, session_dir)
            if json_result["success"]:
                file_results["files_generated"].append(json_result["file_path"])
                session["file_paths"]["json_data"] = str(json_result["file_path"])
            else:
                file_results["errors"].append(f"JSON file generation failed: {json_result['error']}")
            
        except Exception as e:
            file_results["errors"].append(f"File generation error: {str(e)}")
        
        return file_results
    
    def _generate_mot_file(self, session: Dict, session_dir: Path) -> Dict:
        """Generate MOT file from session movement data"""
        try:
            from .file_handlers.mot_handler import MOTFileHandler
            
            mot_handler = MOTFileHandler()
            movement_data = session["movement_data"]
            
            if not movement_data:
                return {"success": False, "error": "No movement data available"}
            
            # Prepare data for MOT file
            time_series = []
            angle_series = []
            
            for data_point in movement_data:
                time_series.append(data_point["timestamp"] - movement_data[0]["timestamp"])  # Relative time
                
                # Extract angle data
                loweback_data = data_point["data"].get("loweback_analysis", {})
                trunk_angle = loweback_data.get("trunk_angle", 0.0)
                angle_series.append(trunk_angle)
            
            # Generate MOT file
            mot_file_path = session_dir / f"{session['session_id']}_motion_data.mot"
            
            mot_data = {
                "time": time_series,
                "lumbar_flexion_extension": angle_series
            }
            
            success = mot_handler.write_mot_file(str(mot_file_path), mot_data)
            
            if success:
                return {
                    "success": True,
                    "file_path": mot_file_path,
                    "data_points": len(time_series)
                }
            else:
                return {"success": False, "error": "MOT file write failed"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_json_file(self, session: Dict, session_dir: Path) -> Dict:
        """Generate JSON file with complete session data"""
        try:
            json_file_path = session_dir / f"{session['session_id']}_complete_data.json"
            
            # Prepare complete session data
            complete_data = {
                "session_info": {
                    "session_id": session["session_id"],
                    "session_type": session["session_type"],
                    "created_at": session["created_at"],
                    "completed_at": session.get("completed_at"),
                    "duration": session.get("completed_at", time.time()) - session["created_at"]
                },
                "analysis_results": session.get("analysis_results"),
                "movement_data": session["movement_data"],
                "file_paths": session["file_paths"]
            }
            
            # Write JSON file
            with open(json_file_path, 'w') as f:
                json.dump(complete_data, f, indent=2, default=str)
            
            return {
                "success": True,
                "file_path": json_file_path,
                "file_size": os.path.getsize(json_file_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        if not self.auto_cleanup:
            return
        
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            session_age = current_time - session["last_activity"]
            if session_age > self.max_session_duration:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
    
    def _cleanup_session(self, session_id: str):
        """Clean up a specific session"""
        if session_id in self.sessions:
            # Don't delete completed sessions with files
            session = self.sessions[session_id]
            if session["status"] == "completed" and session["file_paths"]["mot_file"]:
                return  # Keep completed sessions
            
            # Remove from memory
            del self.sessions[session_id]
            
            # Optionally remove session directory (for incomplete sessions)
            session_dir = self.data_directory / session_id
            if session_dir.exists() and session.get("status") != "completed":
                import shutil
                shutil.rmtree(session_dir, ignore_errors=True)
    
    def get_session_stats(self) -> Dict:
        """Get statistics about all sessions"""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s["status"] == "active")
        completed_sessions = sum(1 for s in self.sessions.values() if s["status"] == "completed")
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "data_directory": str(self.data_directory),
            "auto_cleanup": self.auto_cleanup
        }