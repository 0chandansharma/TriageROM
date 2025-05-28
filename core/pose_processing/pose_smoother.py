#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Optional
from collections import deque
from ...utils.math_utils import MathUtils

class PoseSmoother:
    """Smooth pose landmarks using various methods"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.smoothing_enabled = config.get('enable_smoothing', True)
        self.smoothing_method = config.get('smoothing_method', 'moving_average')
        self.window_size = config.get('smoothing_window_size', 5)
        
        # History storage for each keypoint
        self.keypoint_history = {}
        self.max_history = config.get('max_smoothing_history', 30)
    
    def smooth_pose(self, pose_data: Dict) -> Dict:
        """Apply smoothing to pose data"""
        if not self.smoothing_enabled:
            return pose_data
        
        smoothed_pose = pose_data.copy()
        
        # Smooth spine keypoints
        if "spine_keypoints" in pose_data:
            smoothed_spine = self._smooth_keypoints(pose_data["spine_keypoints"])
            smoothed_pose["spine_keypoints"] = smoothed_spine
        
        # Smooth all pose landmarks
        if "all_pose_landmarks" in pose_data:
            smoothed_landmarks = self._smooth_keypoints(pose_data["all_pose_landmarks"])
            smoothed_pose["all_pose_landmarks"] = smoothed_landmarks
        
        return smoothed_pose
    
    def _smooth_keypoints(self, keypoints: Dict) -> Dict:
        """Smooth individual keypoints"""
        smoothed_keypoints = {}
        
        for kp_name, kp_data in keypoints.items():
            if not isinstance(kp_data, dict):
                smoothed_keypoints[kp_name] = kp_data
                continue
            
            # Initialize history for this keypoint if needed
            if kp_name not in self.keypoint_history:
                self.keypoint_history[kp_name] = {
                    'x': deque(maxlen=self.max_history),
                    'y': deque(maxlen=self.max_history),
                    'z': deque(maxlen=self.max_history)
                }
            
            # Add current values to history
            history = self.keypoint_history[kp_name]
            history['x'].append(kp_data.get('x', 0.0))
            history['y'].append(kp_data.get('y', 0.0))
            history['z'].append(kp_data.get('z', 0.0))
            
            # Apply smoothing
            smoothed_data = kp_data.copy()
            
            if len(history['x']) >= 3:  # Need minimum history for smoothing
                smoothed_data['x'] = self._apply_smoothing(list(history['x']))
                smoothed_data['y'] = self._apply_smoothing(list(history['y']))
                smoothed_data['z'] = self._apply_smoothing(list(history['z']))
                
                # Update pixel coordinates if they exist
                if 'pixel_x' in kp_data and 'pixel_y' in kp_data:
                    # Assume standard image size for conversion
                    img_width = self.config.get('image_width', 640)
                    img_height = self.config.get('image_height', 480)
                    
                    smoothed_data['pixel_x'] = int(smoothed_data['x'] * img_width)
                    smoothed_data['pixel_y'] = int(smoothed_data['y'] * img_height)
            
            smoothed_keypoints[kp_name] = smoothed_data
        
        return smoothed_keypoints
    
    def _apply_smoothing(self, data: List[float]) -> float:
        """Apply smoothing algorithm to data sequence"""
        if len(data) < 2:
            return data[-1] if data else 0.0
        
        try:
            if self.smoothing_method == 'moving_average':
                return self._moving_average_smooth(data)
            elif self.smoothing_method == 'exponential':
                return self._exponential_smooth(data)
            elif self.smoothing_method == 'kalman':
                return self._kalman_smooth(data)
            else:
                return data[-1]  # No smoothing
                
        except Exception as e:
            print(f"Smoothing error: {e}")
            return data[-1]
    
    def _moving_average_smooth(self, data: List[float]) -> float:
        """Moving average smoothing"""
        window_size = min(self.window_size, len(data))
        recent_data = data[-window_size:]
        
        # Remove NaN values
        valid_data = [x for x in recent_data if not np.isnan(x)]
        
        if valid_data:
            return np.mean(valid_data)
        else:
            return data[-1]
    
    def _exponential_smooth(self, data: List[float]) -> float:
        """Exponential smoothing"""
        alpha = self.config.get('exponential_alpha', 0.3)
        
        if len(data) < 2:
            return data[-1]
        
        # Simple exponential smoothing
        smoothed = data[0]
        for value in data[1:]:
            if not np.isnan(value):
                smoothed = alpha * value + (1 - alpha) * smoothed
        
        return smoothed
    
    def _kalman_smooth(self, data: List[float]) -> float:
        """Simple 1D Kalman filter smoothing"""
        if len(data) < 2:
            return data[-1]
        
        # Simple Kalman filter parameters
        process_variance = self.config.get('kalman_process_var', 1e-3)
        measurement_variance = self.config.get('kalman_measurement_var', 1e-1)
        
        # Initialize
        estimate = data[0]
        error_estimate = 1.0
        
        for measurement in data[1:]:
            if np.isnan(measurement):
                continue
            
            # Prediction step
            prediction = estimate
            prediction_error = error_estimate + process_variance
            
            # Update step
            kalman_gain = prediction_error / (prediction_error + measurement_variance)
            estimate = prediction + kalman_gain * (measurement - prediction)
            error_estimate = (1 - kalman_gain) * prediction_error
        
        return estimate
    
    def reset_history(self):
        """Reset smoothing history"""
        self.keypoint_history.clear()
    
    def get_smoothing_info(self) -> Dict:
        """Get smoothing configuration information"""
        return {
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_method": self.smoothing_method,
            "window_size": self.window_size,
            "keypoints_tracked": len(self.keypoint_history),
            "max_history": self.max_history
        }