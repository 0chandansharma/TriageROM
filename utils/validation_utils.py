#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Any, Optional, Tuple
import re

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def validate_pose_data(pose_data: Dict) -> Dict:
        """
        Validate pose estimation data structure
        
        Args:
            pose_data: Pose data dictionary
            
        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ["pose_detected", "spine_keypoints"]
        for field in required_fields:
            if field not in pose_data:
                validation_result["errors"].append(f"Missing required field: {field}")
                validation_result["valid"] = False
        
        # Validate spine keypoints if present
        if "spine_keypoints" in pose_data:
            spine_validation = ValidationUtils.validate_spine_keypoints(
                pose_data["spine_keypoints"]
            )
            if not spine_validation["valid"]:
                validation_result["errors"].extend(spine_validation["errors"])
                validation_result["valid"] = False
            validation_result["warnings"].extend(spine_validation["warnings"])
        
        return validation_result
    
    @staticmethod
    def validate_spine_keypoints(keypoints: Dict) -> Dict:
        """
        Validate spine keypoints data
        
        Args:
            keypoints: Spine keypoints dictionary
            
        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        required_keypoints = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        
        for kp_name in required_keypoints:
            if kp_name not in keypoints:
                validation_result["warnings"].append(f"Missing keypoint: {kp_name}")
                continue
            
            kp_data = keypoints[kp_name]
            
            # Check required fields for each keypoint
            required_kp_fields = ["x", "y", "visibility"]
            for field in required_kp_fields:
                if field not in kp_data:
                    validation_result["errors"].append(f"Missing field '{field}' in keypoint '{kp_name}'")
                    validation_result["valid"] = False
            
            # Validate coordinate ranges
            if "x" in kp_data and not (0 <= kp_data["x"] <= 1):
                validation_result["warnings"].append(f"Keypoint '{kp_name}' x coordinate out of range [0,1]: {kp_data['x']}")
            
            if "y" in kp_data and not (0 <= kp_data["y"] <= 1):
                validation_result["warnings"].append(f"Keypoint '{kp_name}' y coordinate out of range [0,1]: {kp_data['y']}")
            
            # Validate visibility
            if "visibility" in kp_data and not (0 <= kp_data["visibility"] <= 1):
                validation_result["warnings"].append(f"Keypoint '{kp_name}' visibility out of range [0,1]: {kp_data['visibility']}")
        
        return validation_result
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """
        Validate session ID format
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if valid, False otherwise
        """
        if not session_id or not isinstance(session_id, str):
            return False
        
        # Allow alphanumeric characters, underscores, and hyphens
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, session_id)) and len(session_id) >= 8
    
    @staticmethod
    def validate_angle_range(angle: float, angle_type: str) -> bool:
        """
        Validate angle value ranges
        
        Args:
            angle: Angle value in degrees
            angle_type: Type of angle (trunk, hip, etc.)
            
        Returns:
            True if valid, False otherwise
        """
        if angle_type == "trunk":
            return -90 <= angle <= 45  # Reasonable trunk angle range
        elif angle_type == "hip":
            return -30 <= angle <= 150  # Hip flexion range
        else:
            return -180 <= angle <= 180  # General angle range
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing invalid characters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename