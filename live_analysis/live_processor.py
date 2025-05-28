#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
from typing import Dict, Optional, Callable
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from live_analysis.camera_manager import CameraManager
from api.services.live_pose_service import LivePoseService
from api.services.loweback_analysis_service import LowerBackAnalysisService

class LiveProcessor:
    """Process live camera feed for real-time ROM analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.camera_manager = CameraManager(
            camera_id=config.get('camera', {}).get('camera_id', 0),
            config=config.get('camera', {})
        )
        
        self.pose_service = LivePoseService(config)
        self.loweback_service = LowerBackAnalysisService(config)
        
        # Processing state
        self.is_processing = False
        self.session_id = None
        self.frame_callback = None
        self.result_callback = None
        
        # Statistics
        self.total_frames_processed = 0
        self.successful_analyses = 0
        self.processing_times = []
        
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize camera
            if not self.camera_manager.initialize_camera():
                return False
            
            # Initialize pose service
            if not self.pose_service.initialize():
                return False
            
            return True
            
        except Exception as e:
            print(f"Live processor initialization failed: {e}")
            return False
    
    def start_processing(self, session_id: str, 
                        frame_callback: Optional[Callable] = None,
                        result_callback: Optional[Callable] = None) -> bool:
        """Start live processing"""
        try:
            if self.is_processing:
                return False
            
            self.session_id = session_id
            self.frame_callback = frame_callback
            self.result_callback = result_callback
            
            # Start camera capture
            if not self.camera_manager.start_capture():
                return False
            
            # Start analysis session
            if not self.loweback_service.start_session(session_id):
                return False
            
            self.is_processing = True
            return True
            
        except Exception as e:
            print(f"Failed to start live processing: {e}")
            return False
    
    def process_frame(self) -> Optional[Dict]:
        """Process a single frame from camera"""
        if not self.is_processing:
            return None
        
        start_time = time.time()
        
        try:
            # Get frame from camera
            frame = self.camera_manager.get_frame()
            if frame is None:
                return {"error": "No frame available from camera"}
            
            # Optional frame callback (for display, saving, etc.)
            if self.frame_callback:
                self.frame_callback(frame)
            
            # Analyze pose
            pose_result = self.pose_service.analyze_frame(frame)
            
            if not pose_result.get("pose_detected", False):
                return {
                    "status": "no_pose_detected",
                    "message": "No person detected in frame",
                    "timestamp": time.time(),
                    "session_id": self.session_id
                }
            
            # Analyze lower back ROM
            rom_result = self.loweback_service.analyze_movement(
                pose_result, 
                self.session_id
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.total_frames_processed += 1
            self.successful_analyses += 1
            self.processing_times.append(processing_time)
            
            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            # Add processing metadata
            rom_result["processing_time_ms"] = round(processing_time * 1000, 2)
            rom_result["frame_number"] = self.total_frames_processed
            
            # Optional result callback
            if self.result_callback:
                self.result_callback(rom_result)
            
            return rom_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error_type": "PROCESSING_ERROR",
                "message": str(e),
                "timestamp": time.time(),
                "session_id": self.session_id
            }
            
            if self.result_callback:
                self.result_callback(error_result)
            
            return error_result
    
    def stop_processing(self) -> Dict:
        """Stop live processing and get final results"""
        try:
            self.is_processing = False
            
            # Stop camera capture
            self.camera_manager.stop_capture()
            
            # Get final analysis results
            final_results = None
            if self.session_id:
                final_results = self.loweback_service.end_session(self.session_id)
            
            return {
                "status": "stopped",
                "session_id": self.session_id,
                "final_results": final_results,
                "processing_stats": self.get_processing_stats()
            }
            
        except Exception as e:
            return {"error": f"Failed to stop processing: {str(e)}"}
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        avg_processing_time = (
            np.mean(self.processing_times) * 1000 
            if self.processing_times else 0
        )
        
        success_rate = (
            (self.successful_analyses / max(1, self.total_frames_processed)) * 100
        )
        
        return {
            "total_frames_processed": self.total_frames_processed,
            "successful_analyses": self.successful_analyses,
            "success_rate_percent": round(success_rate, 2),
            "average_processing_time_ms": round(avg_processing_time, 2),
            "camera_info": self.camera_manager.get_camera_info(),
            "pose_service_stats": self.pose_service.get_service_stats(),
            "loweback_service_stats": self.loweback_service.get_service_stats()
        }
    
    def cleanup(self):
        """Cleanup all resources"""
        self.stop_processing()
        self.camera_manager.cleanup()
        self.pose_service.cleanup()