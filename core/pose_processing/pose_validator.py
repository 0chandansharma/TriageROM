#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Optional, Tuple

class PoseValidator:
    """Validate pose estimation results"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_pose_confidence = config.get('min_pose_confidence', 0.5)
        self.min_keypoint_visibility = config.get('min_keypoint_visibility', 0.5)
        self.required_keypoints = config.get('required_keypoints', [])
        
        # Validation statistics
        self.total_validations = 0
        self.successful_validations = 0
        self.validation_errors = []
    
    def validate_pose(self, pose_data: Dict) -> Dict:
        """
        Validate pose estimation results
        
        Returns:
            Dictionary with validation results
        """
        self.total_validations += 1
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "confidence_score": 0.0,
            "quality_metrics": {}
        }
        
        try:
            # Check if pose was detected
            if not pose_data.get("pose_detected", False):
                validation_result["valid"] = False
                validation_result["errors"].append("No pose detected")
                return validation_result
            
            # Validate spine keypoints (most important for lower back analysis)
            spine_validation = self._validate_spine_keypoints(
                pose_data.get("spine_keypoints", {})
            )
            
            if not spine_validation["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(spine_validation["errors"])
            
            validation_result["warnings"].extend(spine_validation["warnings"])
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(pose_data)
            validation_result["confidence_score"] = confidence_score
            
            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(pose_data)
            validation_result["quality_metrics"] = quality_metrics
            
            # Check minimum confidence threshold
            if confidence_score < self.min_pose_confidence:
                validation_result["warnings"].append(
                    f"Low pose confidence: {confidence_score:.3f} < {self.min_pose_confidence}"
                )
            
            if validation_result["valid"]:
                self.successful_validations += 1
            else:
                self.validation_errors.extend(validation_result["errors"])
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_spine_keypoints(self, spine_keypoints: Dict) -> Dict:
        """Validate spine keypoints specifically"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        required_spine_keypoints = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        
        # Check presence of required keypoints
        missing_keypoints = []
        low_confidence_keypoints = []
        
        for kp_name in required_spine_keypoints:
            if kp_name not in spine_keypoints:
                missing_keypoints.append(kp_name)
                continue
            
            kp_data = spine_keypoints[kp_name]
            
            # Validate keypoint structure
            if not isinstance(kp_data, dict):
                validation_result["errors"].append(f"Invalid keypoint data structure for {kp_name}")
                validation_result["valid"] = False
                continue
            
            # Check required fields
            required_fields = ["x", "y", "visibility"]
            for field in required_fields:
                if field not in kp_data:
                    validation_result["errors"].append(f"Missing field '{field}' in keypoint '{kp_name}'")
                    validation_result["valid"] = False
            
            # Check coordinate validity
            x, y = kp_data.get("x", 0), kp_data.get("y", 0)
            visibility = kp_data.get("visibility", 0)
            
            if np.isnan(x) or np.isnan(y):
                validation_result["warnings"].append(f"NaN coordinates in keypoint '{kp_name}'")
            
            if not (0 <= x <= 1) or not (0 <= y <= 1):
                validation_result["warnings"].append(f"Coordinates out of range [0,1] for keypoint '{kp_name}': ({x:.3f}, {y:.3f})")
            
            if visibility < self.min_keypoint_visibility:
                low_confidence_keypoints.append((kp_name, visibility))
        
        # Handle missing keypoints
        if missing_keypoints:
            if len(missing_keypoints) >= 3:  # Too many missing
                validation_result["valid"] = False
                validation_result["errors"].append(f"Too many missing keypoints: {missing_keypoints}")
            else:
                validation_result["warnings"].append(f"Missing keypoints: {missing_keypoints}")
        
        # Handle low confidence keypoints
        if low_confidence_keypoints:
            if len(low_confidence_keypoints) >= 3:  # Too many low confidence
                validation_result["valid"] = False
                validation_result["errors"].append(f"Too many low confidence keypoints: {[kp[0] for kp in low_confidence_keypoints]}")
            else:
                validation_result["warnings"].append(f"Low confidence keypoints: {low_confidence_keypoints}")
        
        return validation_result
    
    def _calculate_confidence_score(self, pose_data: Dict) -> float:
        """Calculate overall pose confidence score"""
        try:
            spine_keypoints = pose_data.get("spine_keypoints", {})
            
            if not spine_keypoints:
                return 0.0
            
            # Calculate average visibility of spine keypoints
            visibilities = []
            for kp_data in spine_keypoints.values():
                if isinstance(kp_data, dict) and "visibility" in kp_data:
                    visibility = kp_data["visibility"]
                    if not np.isnan(visibility):
                        visibilities.append(visibility)
            
            if not visibilities:
                return 0.0
            
            return float(np.mean(visibilities))
            
        except Exception as e:
            print(f"Error calculating confidence score: {e}")
            return 0.0
    
    def _calculate_quality_metrics(self, pose_data: Dict) -> Dict:
        """Calculate pose quality metrics"""
        metrics = {
            "keypoint_count": 0,
            "valid_keypoint_count": 0,
            "average_visibility": 0.0,
            "coordinate_validity": 0.0,
            "tracking_stability": 0.0
        }
        
        try:
            spine_keypoints = pose_data.get("spine_keypoints", {})
            
            metrics["keypoint_count"] = len(spine_keypoints)
            
            valid_keypoints = 0
            visibilities = []
            valid_coordinates = 0
            
            for kp_data in spine_keypoints.values():
                if isinstance(kp_data, dict):
                    # Check visibility
                    visibility = kp_data.get("visibility", 0)
                    if not np.isnan(visibility):
                        visibilities.append(visibility)
                        if visibility >= self.min_keypoint_visibility:
                            valid_keypoints += 1
                    
                    # Check coordinate validity
                    x, y = kp_data.get("x", 0), kp_data.get("y", 0)
                    if not (np.isnan(x) or np.isnan(y)) and (0 <= x <= 1) and (0 <= y <= 1):
                        valid_coordinates += 1
            
            metrics["valid_keypoint_count"] = valid_keypoints
            metrics["average_visibility"] = float(np.mean(visibilities)) if visibilities else 0.0
            metrics["coordinate_validity"] = valid_coordinates / max(1, len(spine_keypoints))
            
            # Tracking stability (would need history for proper calculation)
            metrics["tracking_stability"] = 0.8 if pose_data.get("tracking_active", False) else 0.5
            
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
        
        return metrics
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        success_rate = (self.successful_validations / max(1, self.total_validations)) * 100
        
        return {
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "success_rate_percent": round(success_rate, 2),
            "min_pose_confidence": self.min_pose_confidence,
            "min_keypoint_visibility": self.min_keypoint_visibility,
            "required_keypoints": self.required_keypoints,
            "recent_errors": self.validation_errors[-10:] if self.validation_errors else []
        }
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.total_validations = 0
        self.successful_validations = 0
        self.validation_errors.clear()