#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, Tuple, Optional, List
from .angle_calculator import AngleCalculator

class SpineAngleCalculator:
    """Specialized calculator for spine angles, particularly lower back"""
    
    def __init__(self):
        self.angle_calc = AngleCalculator()
        self.previous_angles = []
        
    def calculate_loweback_angles(self, spine_keypoints: Dict) -> Dict:
        """
        Calculate lower back flexion/extension angles
        
        Args:
            spine_keypoints: Dictionary with nose, shoulders, and hips keypoints
            
        Returns:
            Dictionary with calculated angles and metrics
        """
        try:
            # Extract keypoint coordinates
            nose = self._get_keypoint_coords(spine_keypoints, "nose")
            left_shoulder = self._get_keypoint_coords(spine_keypoints, "left_shoulder")
            right_shoulder = self._get_keypoint_coords(spine_keypoints, "right_shoulder")
            left_hip = self._get_keypoint_coords(spine_keypoints, "left_hip")
            right_hip = self._get_keypoint_coords(spine_keypoints, "right_hip")
            
            if None in [nose, left_shoulder, right_shoulder, left_hip, right_hip]:
                return self._create_error_result("Missing required keypoints")
            
            # Calculate midpoints
            shoulder_mid = self._calculate_midpoint(left_shoulder, right_shoulder)
            hip_mid = self._calculate_midpoint(left_hip, right_hip)
            
            # Calculate trunk angle (primary measurement)
            trunk_angle = self.angle_calc.calculate_trunk_angle(shoulder_mid, hip_mid)
            
            # Calculate hip flexion to detect compensation
            # Use average of left and right hip angles
            left_knee = self._estimate_knee_position(left_hip, left_shoulder)
            right_knee = self._estimate_knee_position(right_hip, right_shoulder)
            
            left_hip_angle = self.angle_calc.calculate_hip_angle(left_hip, left_knee, shoulder_mid)
            right_hip_angle = self.angle_calc.calculate_hip_angle(right_hip, right_knee, shoulder_mid)
            
            hip_angle = np.nanmean([left_hip_angle, right_hip_angle])
            
            # Calculate net spine movement (compensating for hip movement)
            net_spine_angle = self._calculate_net_spine_movement(trunk_angle, hip_angle)
            
            # Determine movement direction and phase
            movement_info = self._analyze_movement_pattern(trunk_angle)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(spine_keypoints, trunk_angle)
            
            result = {
                "trunk_angle": round(trunk_angle, 2) if not np.isnan(trunk_angle) else None,
                "hip_angle": round(hip_angle, 2) if not np.isnan(hip_angle) else None,
                "net_spine_angle": round(net_spine_angle, 2) if not np.isnan(net_spine_angle) else None,
                "movement_direction": movement_info["direction"],
                "movement_phase": movement_info["phase"],
                "quality_metrics": quality_metrics,
                "calculation_successful": True
            }
            
            # Store for movement analysis
            if not np.isnan(trunk_angle):
                self.previous_angles.append(trunk_angle)
                # Keep only last 30 angles for analysis
                if len(self.previous_angles) > 30:
                    self.previous_angles.pop(0)
            
            return result
            
        except Exception as e:
            return self._create_error_result(f"Calculation error: {str(e)}")
    
    def get_rom_analysis(self) -> Dict:
        """Analyze range of motion from stored angle history"""
        if len(self.previous_angles) < 5:
            return {
                "insufficient_data": True,
                "message": "Need more movement data for ROM analysis"
            }
        
        valid_angles = [a for a in self.previous_angles if not np.isnan(a)]
        
        if not valid_angles:
            return {"insufficient_data": True, "message": "No valid angles recorded"}
        
        max_flexion = min(valid_angles)  # Most negative (forward bend)
        max_extension = max(valid_angles)  # Most positive (backward bend)
        total_rom = max_extension - max_flexion
        
        # Detect movement patterns
        movement_velocity = self._calculate_movement_velocity(valid_angles)
        smoothness_score = self._calculate_smoothness(valid_angles)
        
        return {
            "max_flexion": round(max_flexion, 2),
            "max_extension": round(max_extension, 2),
            "total_rom": round(total_rom, 2),
            "current_angle": round(valid_angles[-1], 2),
            "movement_velocity": round(movement_velocity, 2),
            "smoothness_score": round(smoothness_score, 3),
            "data_points": len(valid_angles),
            "sufficient_data": True
        }
    
    def _get_keypoint_coords(self, keypoints: Dict, keypoint_name: str) -> Optional[Tuple[float, float]]:
        """Extract x, y coordinates from keypoint data"""
        if keypoint_name in keypoints:
            kp = keypoints[keypoint_name]
            if kp.get('visibility', 0) > 0.5:  # Sufficient visibility
                return (kp['x'], kp['y'])
        return None
    
    def _calculate_midpoint(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate midpoint between two points"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    def _estimate_knee_position(self, hip: Tuple[float, float], shoulder: Tuple[float, float]) -> Tuple[float, float]:
        """Estimate knee position for hip angle calculation when knee not visible"""
        # Simple estimation: knee is below hip in vertical line
        # This is approximate for standing position
        return (hip[0], hip[1] + 0.3)  # 0.3 is approximate leg length ratio
    
    def _calculate_net_spine_movement(self, trunk_angle: float, hip_angle: float) -> float:
        """Calculate spine movement compensating for hip contribution"""
        if np.isnan(trunk_angle) or np.isnan(hip_angle):
            return trunk_angle
        
        # Simple compensation model - can be refined based on biomechanical research
        hip_compensation_factor = 0.3  # Hip contributes ~30% to forward bending
        compensated_angle = trunk_angle - (hip_angle * hip_compensation_factor)
        
        return compensated_angle
    
    def _analyze_movement_pattern(self, current_angle: float) -> Dict:
        """Analyze movement direction and phase"""
        if len(self.previous_angles) < 3:
            return {"direction": "unknown", "phase": "unknown"}
        
        recent_angles = self.previous_angles[-3:]
        angle_changes = np.diff(recent_angles)
        
        if np.mean(angle_changes) > 1:
            direction = "extending"
        elif np.mean(angle_changes) < -1:
            direction = "flexing"
        else:
            direction = "holding"
        
        # Determine phase based on angle magnitude
        if abs(current_angle) < 5:
            phase = "neutral"
        elif current_angle < -20:
            phase = "deep_flexion"
        elif current_angle > 10:
            phase = "extension"
        else:
            phase = "mid_range"
        
        return {"direction": direction, "phase": phase}
    
    def _calculate_quality_metrics(self, keypoints: Dict, trunk_angle: float) -> Dict:
        """Calculate movement quality metrics"""
        # Pose stability (based on keypoint confidence)
        confidences = []
        for kp_name in ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
            if kp_name in keypoints:
                confidences.append(keypoints[kp_name].get('visibility', 0))
        
        pose_stability = np.mean(confidences) if confidences else 0
        
        # Movement smoothness (based on angle history)
        smoothness = 0.5  # Default
        if len(self.previous_angles) > 5:
            angle_velocities = np.diff(self.previous_angles[-6:])
            smoothness = 1.0 - (np.std(angle_velocities) / 10.0)  # Normalize by expected std
            smoothness = max(0, min(1, smoothness))
        
        # Overall confidence
        confidence_score = (pose_stability + smoothness) / 2
        
        return {
            "pose_stability": round(pose_stability, 3),
            "movement_smoothness": round(smoothness, 3),
            "confidence_score": round(confidence_score, 3)
        }
    
    def _calculate_movement_velocity(self, angles: List[float]) -> float:
        """Calculate average movement velocity"""
        if len(angles) < 2:
            return 0.0
        
        velocities = np.diff(angles)
        return float(np.mean(np.abs(velocities)))
    
    def _calculate_smoothness(self, angles: List[float]) -> float:
        """Calculate movement smoothness score (0-1, higher is smoother)"""
        if len(angles) < 3:
            return 0.5
        
        velocities = np.diff(angles)
        accelerations = np.diff(velocities)
        
        # Smoothness inversely related to acceleration variance
        if len(accelerations) > 0:
            smoothness = 1.0 / (1.0 + np.var(accelerations))
            return min(1.0, smoothness)
        
        return 0.5
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create standardized error result"""
        return {
            "trunk_angle": None,
            "hip_angle": None,
            "net_spine_angle": None,
            "movement_direction": "unknown",
            "movement_phase": "unknown",
            "quality_metrics": {
                "pose_stability": 0.0,
                "movement_smoothness": 0.0,
                "confidence_score": 0.0
            },
            "calculation_successful": False,
            "error": error_message
        }
    
    def reset_history(self):
        """Reset angle history for new session"""
        self.previous_angles = []