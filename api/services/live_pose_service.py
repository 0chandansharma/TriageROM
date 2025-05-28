#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from typing import Dict, Optional
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.pose_estimation.model_factory import PoseModelFactory
from core.pose_estimation.mediapipe_holistic import MediaPipeHolisticEstimator
from core.pose_processing.pose_tracker import PoseTracker
from core.pose_processing.pose_filter import PoseFilter
from utils.validation_utils import ValidationUtils

class LivePoseService:
    """Service for live pose estimation and processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pose_estimator = None
        self.pose_tracker = PoseTracker()
        self.pose_filter = PoseFilter(config.get("processing", {}))
        
        # Model selection
        self.model_type = config.get("pose", {}).get("model_type", "mediapipe")
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Quality tracking
        self.successful_detections = 0
        self.failed_detections = 0
        
    def initialize(self) -> bool:
        """Initialize the pose estimation service"""
        try:
            # Create pose estimator using factory
            self.pose_estimator = PoseModelFactory.create_estimator(
                self.model_type, 
                self.config.get("pose", {})
            )
            
            if self.pose_estimator is None:
                print(f"Failed to create pose estimator for model type: {self.model_type}")
                return False
            
            success = self.pose_estimator.initialize()
            if not success:
                print(f"Failed to initialize {self.model_type} pose estimator")
                return False
            
            print(f"Successfully initialized {self.model_type} pose estimator")
            return True
            
        except Exception as e:
            print(f"Failed to initialize pose service: {e}")
            return False
    
    def analyze_frame(self, image: np.ndarray) -> Dict:
        """
        Analyze a single frame for pose estimation
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Pose analysis results
        """
        start_time = time.time()
        
        try:
            # Validate input
            if image is None or image.size == 0:
                return self._create_error_result("Invalid input image")
            
            # Ensure image is in correct format
            if len(image.shape) != 3 or image.shape[2] != 3:
                return self._create_error_result("Image must be 3-channel BGR format")
            
            # Estimate pose
            pose_result = self.pose_estimator.estimate_pose(image)
            
            if not pose_result.get("processing_successful", False):
                self.failed_detections += 1
                return self._create_error_result("Pose estimation failed")
            
            # Track pose across frames
            tracked_result = self.pose_tracker.track_pose(pose_result)
            
            # Apply filtering if enabled
            if self.config.get("processing", {}).get("enable_filtering", True):
                filtered_result = self.pose_filter.filter_pose(tracked_result)
            else:
                filtered_result = tracked_result
            
            # Validate results
            validation_result = ValidationUtils.validate_pose_data(filtered_result)
            
            if not validation_result["valid"]:
                self.failed_detections += 1
                return self._create_error_result(
                    f"Pose validation failed: {'; '.join(validation_result['errors'])}"
                )
            
            # Update statistics
            self.successful_detections += 1
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            # Add metadata
            filtered_result.update({
                "processing_time_ms": round(processing_time * 1000, 2),
                "timestamp": time.time(),
                "frame_number": self.frame_count,
                "service_stats": {
                    "current_fps": self.current_fps,
                    "success_rate": self._calculate_success_rate()
                }
            })
            
            return filtered_result
            
        except Exception as e:
            self.failed_detections += 1
            return self._create_error_result(f"Processing error: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create standardized error result"""
        return {
            "pose_detected": False,
            "processing_successful": False,
            "error": error_message,
            "timestamp": time.time(),
            "frame_number": self.frame_count,
            "spine_keypoints": {},
            "all_pose_landmarks": {},
            "service_stats": {
                "current_fps": self.current_fps,
                "success_rate": self._calculate_success_rate()
            }
        }
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.frame_count += 1
        self.total_processing_time += processing_time
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            time_elapsed = current_time - self.last_fps_update
            frames_in_period = 1  # Approximate for this update period
            self.current_fps = frames_in_period / time_elapsed
            self.last_fps_update = current_time
    
    def _calculate_success_rate(self) -> float:
        """Calculate pose detection success rate"""
        total_attempts = self.successful_detections + self.failed_detections
        if total_attempts == 0:
            return 0.0
        return (self.successful_detections / total_attempts) * 100
    
    def is_ready(self) -> bool:
        """Check if service is ready for processing"""
        return (self.pose_estimator is not None and 
                self.pose_estimator.is_initialized)
    
    def get_service_stats(self) -> Dict:
        """Get service statistics"""
        avg_processing_time = (self.total_processing_time / max(1, self.frame_count)) * 1000
        
        return {
            "service_ready": self.is_ready(),
            "frames_processed": self.frame_count,
            "successful_detections": self.successful_detections,
            "failed_detections": self.failed_detections,
            "success_rate_percent": round(self._calculate_success_rate(), 2),
            "current_fps": round(self.current_fps, 2),
            "average_processing_time_ms": round(avg_processing_time, 2),
            "pose_tracker_stats": self.pose_tracker.get_tracking_stats(),
            "filter_info": self.pose_filter.get_filter_info()
        }
    
    def reset_stats(self):
        """Reset service statistics"""
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.successful_detections = 0
        self.failed_detections = 0
        self.current_fps = 0.0
        self.last_fps_update = time.time()
    
    def cleanup(self):
        """Cleanup service resources"""
        if self.pose_estimator:
            self.pose_estimator.cleanup()
        
        self.pose_tracker.reset_tracking()
        self.pose_filter.reset_history()