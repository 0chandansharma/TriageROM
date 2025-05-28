#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import Dict, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rom_analysis.spine.lumbar.lumbar_rom_analyzer import LumbarROMAnalyzer
from utils.validation_utils import ValidationUtils

class LowerBackAnalysisService:
    """Service for lower back ROM analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.active_sessions = {}
        self.session_results = {}
        
        # Service statistics
        self.total_analyses = 0
        self.successful_analyses = 0
        self.sessions_created = 0
        self.sessions_completed = 0
    
    def analyze_movement(self, pose_data: Dict, session_id: str) -> Dict:
        """
        Analyze lower back movement from pose data
        
        Args:
            pose_data: Pose estimation results
            session_id: Session identifier
            
        Returns:
            Movement analysis results
        """
        try:
            self.total_analyses += 1
            
            # Get or create ROM analyzer for session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = LumbarROMAnalyzer(
                    self.config.get("loweback", {})
                )
                self.sessions_created += 1
            
            analyzer = self.active_sessions[session_id]
            
            # Validate pose data
            validation_result = ValidationUtils.validate_pose_data(pose_data)
            if not validation_result["valid"]:
                return self._create_error_response(
                    f"Invalid pose data: {'; '.join(validation_result['errors'])}",
                    session_id
                )
            
            # Analyze movement
            analysis_result = analyzer.analyze_movement(pose_data)
            
            # Check if movement is complete
            if analyzer.detect_movement_completion():
                # Store final results
                self.session_results[session_id] = analyzer.get_rom_summary()
                self.sessions_completed += 1
                
                # Add completion flag to response
                analysis_result["movement_completed"] = True
                analysis_result["final_summary"] = self.session_results[session_id]
            else:
                analysis_result["movement_completed"] = False
            
            self.successful_analyses += 1
            return analysis_result
            
        except Exception as e:
            return self._create_error_response(f"Analysis error: {str(e)}", session_id)
    
    def get_final_results(self, session_id: str) -> Optional[Dict]:
        """Get final analysis results for a session"""
        if session_id in self.session_results:
            return self.session_results[session_id]
        
        # If no stored results, try to get from active analyzer
        if session_id in self.active_sessions:
            analyzer = self.active_sessions[session_id]
            return analyzer.get_rom_summary()
        
        return None
    
    def start_session(self, session_id: str) -> bool:
        """Start a new analysis session"""
        try:
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = LumbarROMAnalyzer(
                    self.config.get("loweback", {})
                )
                self.sessions_created += 1
            
            self.active_sessions[session_id].start_session()
            return True
            
        except Exception as e:
            print(f"Failed to start session {session_id}: {e}")
            return False
    
    def end_session(self, session_id: str) -> Dict:
        """End an analysis session and get final results"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            analyzer = self.active_sessions[session_id]
            analyzer.end_session()
            
            # Get final results
            final_results = analyzer.get_rom_summary()
            self.session_results[session_id] = final_results
            
            # Clean up active session
            del self.active_sessions[session_id]
            self.sessions_completed += 1
            
            return final_results
            
        except Exception as e:
            return {"error": f"Failed to end session: {str(e)}"}
    
    def reset_session(self, session_id: str) -> bool:
        """Reset an analysis session"""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].reset_analysis()
                return True
            return False
            
        except Exception as e:
            print(f"Failed to reset session {session_id}: {e}")
            return False
    
    def get_session_status(self, session_id: str) -> Dict:
        """Get status of an analysis session"""
        if session_id not in self.active_sessions:
            return {
                "exists": False,
                "status": "not_found"
            }
        
        analyzer = self.active_sessions[session_id]
        session_info = analyzer.get_session_info()
        
        return {
            "exists": True,
            "status": "active" if session_info["session_active"] else "inactive",
            "session_info": session_info,
            "has_final_results": session_id in self.session_results
        }
    
    def _create_error_response(self, error_message: str, session_id: str) -> Dict:
        """Create standardized error response"""
        return {
            "status": "error",
            "error_type": "ANALYSIS_ERROR",
            "message": error_message,
            "timestamp": time.time(),
            "session_id": session_id,
            "session_active": session_id in self.active_sessions
        }
    
    def get_service_stats(self) -> Dict:
        """Get service statistics"""
        success_rate = (self.successful_analyses / max(1, self.total_analyses)) * 100
        
        return {
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "success_rate_percent": round(success_rate, 2),
            "active_sessions": len(self.active_sessions),
            "sessions_created": self.sessions_created,
            "sessions_completed": self.sessions_completed,
            "completed_sessions_with_results": len(self.session_results)
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Keep results for a while, could implement cleanup based on age
        # if session_id in self.session_results:
        #     del self.session_results[session_id]