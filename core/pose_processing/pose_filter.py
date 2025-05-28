#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Tuple
from collections import deque

class PoseFilter:
    """
    Pose filtering and smoothing adapted from Sports2D
    """
    
    def __init__(self, filter_config: Dict):
        self.filter_config = filter_config
        self.filter_type = filter_config.get('filter_type', 'butterworth')
        self.enable_filtering = filter_config.get('enable_filtering', True)
        
        # Butterworth filter parameters
        self.filter_order = filter_config.get('filter_order', 4)
        self.cutoff_frequency = filter_config.get('filter_cutoff', 6.0)
        self.sampling_rate = filter_config.get('sampling_rate', 30.0)
        
        # History for filtering
        self.keypoint_history = {}
        self.max_history = 60  # 2 seconds at 30fps
        
    def filter_pose(self, pose_data: Dict) -> Dict:
        """
        Apply filtering to pose data
        
        Args:
            pose_data: Raw pose detection results
            
        Returns:
            Filtered pose data
        """
        if not self.enable_filtering:
            return pose_data
        
        if not pose_data.get("pose_detected", False):
            return pose_data
        
        filtered_pose = pose_data.copy()
        
        # Filter spine keypoints
        if "spine_keypoints" in pose_data:
            filtered_spine = self._filter_keypoints(pose_data["spine_keypoints"])
            filtered_pose["spine_keypoints"] = filtered_spine
        
        # Filter all pose landmarks if available
        if "all_pose_landmarks" in pose_data:
            filtered_landmarks = self._filter_keypoints(pose_data["all_pose_landmarks"])
            filtered_pose["all_pose_landmarks"] = filtered_landmarks
        
        return filtered_pose
    
    def _filter_keypoints(self, keypoints: Dict) -> Dict:
        """Filter individual keypoints"""
        filtered_keypoints = {}
        
        for kp_name, kp_data in keypoints.items():
            if not isinstance(kp_data, dict) or 'x' not in kp_data or 'y' not in kp_data:
                filtered_keypoints[kp_name] = kp_data
                continue
            
            # Store keypoint history
            if kp_name not in self.keypoint_history:
                self.keypoint_history[kp_name] = {
                    'x': deque(maxlen=self.max_history),
                    'y': deque(maxlen=self.max_history),
                    'z': deque(maxlen=self.max_history)
                }
            
            # Add current values to history
            self.keypoint_history[kp_name]['x'].append(kp_data['x'])
            self.keypoint_history[kp_name]['y'].append(kp_data['y'])
            self.keypoint_history[kp_name]['z'].append(kp_data.get('z', 0.0))
            
            # Apply filtering
            filtered_data = kp_data.copy()
            
            if len(self.keypoint_history[kp_name]['x']) >= 5:  # Need minimum history
                filtered_x = self._apply_filter(list(self.keypoint_history[kp_name]['x']))
                filtered_y = self._apply_filter(list(self.keypoint_history[kp_name]['y']))
                filtered_z = self._apply_filter(list(self.keypoint_history[kp_name]['z']))
                
                if filtered_x is not None:
                    filtered_data['x'] = filtered_x[-1]  # Take most recent filtered value
                if filtered_y is not None:
                    filtered_data['y'] = filtered_y[-1]
                if filtered_z is not None and 'z' in kp_data:
                    filtered_data['z'] = filtered_z[-1]
                
                # Update pixel coordinates if they exist
                if 'pixel_x' in kp_data and 'pixel_y' in kp_data:
                    # Assume image dimensions for conversion (this should be passed in config)
                    img_width = self.filter_config.get('image_width', 640)
                    img_height = self.filter_config.get('image_height', 480)
                    
                    filtered_data['pixel_x'] = int(filtered_data['x'] * img_width)
                    filtered_data['pixel_y'] = int(filtered_data['y'] * img_height)
            
            filtered_keypoints[kp_name] = filtered_data
        
        return filtered_keypoints
    
    def _apply_filter(self, data: List[float]) -> Optional[List[float]]:
        """Apply the configured filter to data sequence"""
        if len(data) < 5:  # Need minimum data points
            return None
        
        try:
            # Remove NaN values but keep track of positions
            valid_indices = []
            valid_data = []
            
            for i, val in enumerate(data):
                if not np.isnan(val):
                    valid_indices.append(i)
                    valid_data.append(val)
            
            if len(valid_data) < 5:
                return None
            
            # Apply filter based on type
            if self.filter_type == 'butterworth':
                filtered_data = self._butterworth_filter(valid_data)
            elif self.filter_type == 'gaussian':
                filtered_data = self._gaussian_filter(valid_data)
            elif self.filter_type == 'median':
                filtered_data = self._median_filter(valid_data)
            else:
                return data  # No filtering
            
            if filtered_data is None:
                return data
            
            # Reconstruct full array with NaN values in original positions
            result = [np.nan] * len(data)
            for i, idx in enumerate(valid_indices):
                if i < len(filtered_data):
                    result[idx] = filtered_data[i]
            
            return result
            
        except Exception as e:
            print(f"Filtering error: {e}")
            return data
    
    def _butterworth_filter(self, data: List[float]) -> Optional[List[float]]:
        """Apply Butterworth low-pass filter"""
        try:
            # Design the filter
            nyquist = self.sampling_rate / 2
            normal_cutoff = self.cutoff_frequency / nyquist
            
            if normal_cutoff >= 1.0:
                return data  # Cutoff too high
            
            b, a = signal.butter(self.filter_order, normal_cutoff, btype='low', analog=False)
            
            # Apply filter (using filtfilt for zero-phase filtering)
            filtered = signal.filtfilt(b, a, data)
            
            return filtered.tolist()
            
        except Exception:
            return None
    
    def _gaussian_filter(self, data: List[float]) -> Optional[List[float]]:
        """Apply Gaussian smoothing filter"""
        try:
            from scipy.ndimage import gaussian_filter1d
            sigma = self.filter_config.get('gaussian_sigma', 1.0)
            filtered = gaussian_filter1d(data, sigma=sigma)
            return filtered.tolist()
        except Exception:
            return None
    
    def _median_filter(self, data: List[float]) -> Optional[List[float]]:
        """Apply median filter"""
        try:
            kernel_size = self.filter_config.get('median_kernel_size', 5)
            filtered = signal.medfilt(data, kernel_size=kernel_size)
            return filtered.tolist()
        except Exception:
            return None
    
    def reset_history(self):
        """Reset filter history"""
        self.keypoint_history.clear()
    
    def get_filter_info(self) -> Dict:
        """Get filter configuration information"""
        return {
            "filter_type": self.filter_type,
            "enabled": self.enable_filtering,
            "filter_order": self.filter_order,
            "cutoff_frequency": self.cutoff_frequency,
            "sampling_rate": self.sampling_rate,
            "history_length": len(next(iter(self.keypoint_history.values()), {}).get('x', []))
        }