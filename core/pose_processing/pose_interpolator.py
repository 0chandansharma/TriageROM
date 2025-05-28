#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Optional
from ...utils.common import interpolate_zeros_nans

class PoseInterpolator:
    """Interpolate missing pose data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.interpolation_enabled = config.get('interpolate_missing', True)
        self.max_gap_frames = config.get('max_gap_frames', 10)
        self.interpolation_method = config.get('interpolation_method', 'linear')
        
        # Store pose history for interpolation
        self.pose_history = []
        self.max_history = 50
    
    def interpolate_pose(self, pose_data: Dict) -> Dict:
        """Interpolate missing pose data"""
        if not self.interpolation_enabled:
            return pose_data
        
        # Store current pose in history
        self.pose_history.append(pose_data.copy())
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        # If pose not detected, try to interpolate
        if not pose_data.get("pose_detected", False):
            return self._interpolate_missing_pose()
        
        # If pose detected but some keypoints missing, interpolate those
        return self._interpolate_missing_keypoints(pose_data)
    
    def _interpolate_missing_pose(self) -> Dict:
        """Interpolate when entire pose is missing"""
        if len(self.pose_history) < 3:
            return {
                "pose_detected": False,
                "interpolated": False,
                "error": "Insufficient history for interpolation"
            }
        
        # Find recent valid poses
        valid_poses = [pose for pose in self.pose_history[-10:] 
                      if pose.get("pose_detected", False)]
        
        if len(valid_poses) < 2:
            return {
                "pose_detected": False,
                "interpolated": False,
                "error": "Insufficient valid poses for interpolation"
            }
        
        # Use the most recent valid pose as base
        interpolated_pose = valid_poses[-1].copy()
        interpolated_pose["interpolated"] = True
        interpolated_pose["interpolation_method"] = "last_valid_pose"
        
        return interpolated_pose
    
    def _interpolate_missing_keypoints(self, pose_data: Dict) -> Dict:
        """Interpolate missing keypoints in otherwise valid pose"""
        if len(self.pose_history) < 3:
            return pose_data
        
        interpolated_pose = pose_data.copy()
        
        # Interpolate spine keypoints
        if "spine_keypoints" in pose_data:
            interpolated_spine = self._interpolate_keypoint_dict(
                pose_data["spine_keypoints"], "spine_keypoints"
            )
            interpolated_pose["spine_keypoints"] = interpolated_spine
        
        # Interpolate all pose landmarks
        if "all_pose_landmarks" in pose_data:
            interpolated_landmarks = self._interpolate_keypoint_dict(
                pose_data["all_pose_landmarks"], "all_pose_landmarks"
            )
            interpolated_pose["all_pose_landmarks"] = interpolated_landmarks
        
        return interpolated_pose
    
    def _interpolate_keypoint_dict(self, keypoints: Dict, keypoint_type: str) -> Dict:
        """Interpolate missing keypoints in a keypoint dictionary"""
        interpolated_keypoints = {}
        
        for kp_name, kp_data in keypoints.items():
            if not isinstance(kp_data, dict):
                interpolated_keypoints[kp_name] = kp_data
                continue
            
            # Check if keypoint needs interpolation
            if (kp_data.get('visibility', 0) < 0.3 or 
                np.isnan(kp_data.get('x', 0)) or 
                np.isnan(kp_data.get('y', 0))):
                
                # Try to interpolate from history
                interpolated_kp = self._interpolate_single_keypoint(kp_name, keypoint_type)
                if interpolated_kp:
                    interpolated_kp["interpolated"] = True
                    interpolated_keypoints[kp_name] = interpolated_kp
                else:
                    interpolated_keypoints[kp_name] = kp_data
            else:
                interpolated_keypoints[kp_name] = kp_data
        
        return interpolated_keypoints
    
    def _interpolate_single_keypoint(self, kp_name: str, keypoint_type: str) -> Optional[Dict]:
        """Interpolate a single keypoint from history"""
        # Extract keypoint history
        x_values = []
        y_values = []
        z_values = []
        timestamps = []
        
        for i, pose in enumerate(self.pose_history[-self.max_gap_frames:]):
            if (keypoint_type in pose and 
                kp_name in pose[keypoint_type] and
                isinstance(pose[keypoint_type][kp_name], dict)):
                
                kp = pose[keypoint_type][kp_name]
                if kp.get('visibility', 0) >= 0.3:
                    x_values.append(kp.get('x', np.nan))
                    y_values.append(kp.get('y', np.nan))
                    z_values.append(kp.get('z', np.nan))
                    timestamps.append(i)
                else:
                    x_values.append(np.nan)
                    y_values.append(np.nan)
                    z_values.append(np.nan)
                    timestamps.append(i)
        
        if len([x for x in x_values if not np.isnan(x)]) < 2:
            return None
        
        # Interpolate missing values
        try:
            x_interpolated = interpolate_zeros_nans(x_values, self.max_gap_frames)
            y_interpolated = interpolate_zeros_nans(y_values, self.max_gap_frames)
            z_interpolated = interpolate_zeros_nans(z_values, self.max_gap_frames)
            
            # Return interpolated keypoint (last value)
            return {
                'x': x_interpolated[-1] if x_interpolated else np.nan,
                'y': y_interpolated[-1] if y_interpolated else np.nan,
                'z': z_interpolated[-1] if z_interpolated else np.nan,
                'visibility': 0.7,  # Lower confidence for interpolated
                'interpolated': True
            }
            
        except Exception as e:
            print(f"Interpolation error for {kp_name}: {e}")
            return None
    
    def reset_history(self):
        """Reset interpolation history"""
        self.pose_history.clear()
    
    def get_interpolation_info(self) -> Dict:
        """Get interpolation configuration information"""
        return {
            "interpolation_enabled": self.interpolation_enabled,
            "max_gap_frames": self.max_gap_frames,
            "interpolation_method": self.interpolation_method,
            "history_length": len(self.pose_history),
            "max_history": self.max_history
        }