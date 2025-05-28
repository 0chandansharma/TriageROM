#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from typing import List, Tuple, Optional, Union
from scipy import signal

class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(value, max_val))
    
    @staticmethod
    def normalize_angle(angle_deg: float) -> float:
        """Normalize angle to [-180, 180] range"""
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
        return angle_deg
    
    @staticmethod
    def moving_average(data: List[float], window_size: int) -> List[float]:
        """Calculate moving average with specified window size"""
        if len(data) < window_size:
            return data
        
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            
            window_data = data[start_idx:end_idx]
            valid_data = [x for x in window_data if not np.isnan(x)]
            
            if valid_data:
                result.append(np.mean(valid_data))
            else:
                result.append(np.nan)
        
        return result
    
    @staticmethod
    def calculate_velocity(positions: List[float], time_step: float = 1.0) -> List[float]:
        """Calculate velocity from position data"""
        if len(positions) < 2:
            return [0.0] * len(positions)
        
        velocities = [0.0]  # First velocity is 0
        
        for i in range(1, len(positions)):
            if not np.isnan(positions[i]) and not np.isnan(positions[i-1]):
                velocity = (positions[i] - positions[i-1]) / time_step
                velocities.append(velocity)
            else:
                velocities.append(0.0)
        
        return velocities
    
    @staticmethod
    def calculate_acceleration(velocities: List[float], time_step: float = 1.0) -> List[float]:
        """Calculate acceleration from velocity data"""
        if len(velocities) < 2:
            return [0.0] * len(velocities)
        
        accelerations = [0.0]  # First acceleration is 0
        
        for i in range(1, len(velocities)):
            if not np.isnan(velocities[i]) and not np.isnan(velocities[i-1]):
                acceleration = (velocities[i] - velocities[i-1]) / time_step
                accelerations.append(acceleration)
            else:
                accelerations.append(0.0)
        
        return accelerations
    
    @staticmethod
    def find_peaks_simple(data: List[float], min_height: float = None, min_distance: int = 1) -> List[int]:
        """Find peaks in data with simple criteria"""
        if len(data) < 3:
            return []
        
        peaks = []
        
        for i in range(1, len(data) - 1):
            if np.isnan(data[i]):
                continue
            
            # Check if it's a local maximum
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # Check height threshold
                if min_height is None or data[i] >= min_height:
                    # Check distance from previous peak
                    if not peaks or (i - peaks[-1]) >= min_distance:
                        peaks.append(i)
        
        return peaks
    
    @staticmethod
    def interpolate_linear(x: List[float], y: List[float], x_new: List[float]) -> List[float]:
        """Linear interpolation"""
        if len(x) != len(y) or len(x) < 2:
            return [np.nan] * len(x_new)
        
        y_new = []
        
        for x_val in x_new:
            if x_val <= x[0]:
                y_new.append(y[0])
            elif x_val >= x[-1]:
                y_new.append(y[-1])
            else:
                # Find surrounding points
                for i in range(len(x) - 1):
                    if x[i] <= x_val <= x[i+1]:
                        # Linear interpolation
                        t = (x_val - x[i]) / (x[i+1] - x[i])
                        y_val = y[i] + t * (y[i+1] - y[i])
                        y_new.append(y_val)
                        break
        
        return y_new
    
    @staticmethod
    def calculate_rms(data: List[float]) -> float:
        """Calculate root mean square"""
        valid_data = [x for x in data if not np.isnan(x)]
        if not valid_data:
            return 0.0
        
        return math.sqrt(np.mean([x**2 for x in valid_data]))
    
    @staticmethod
    def calculate_std_dev(data: List[float]) -> float:
        """Calculate standard deviation"""
        valid_data = [x for x in data if not np.isnan(x)]
        if len(valid_data) < 2:
            return 0.0
        
        return float(np.std(valid_data))
    
    @staticmethod
    def smooth_data(data: List[float], method: str = 'moving_average', **kwargs) -> List[float]:
        """Smooth data using specified method"""
        if method == 'moving_average':
            window_size = kwargs.get('window_size', 5)
            return MathUtils.moving_average(data, window_size)
        
        elif method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            valid_indices = [i for i, x in enumerate(data) if not np.isnan(x)]
            
            if len(valid_indices) < 3:
                return data
            
            # Apply Gaussian filter only to valid data
            valid_data = [data[i] for i in valid_indices]
            smoothed_valid = signal.gaussian_filter1d(valid_data, sigma=sigma)
            
            # Reconstruct full array
            result = data.copy()
            for i, idx in enumerate(valid_indices):
                result[idx] = smoothed_valid[i]
            
            return result
        
        else:
            return data