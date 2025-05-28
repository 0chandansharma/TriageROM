#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Optional
from scipy import interpolate

class GapFiller:
    """Fill gaps in data using various interpolation methods"""
    
    def __init__(self, max_gap_size: int = 10, method: str = 'linear'):
        self.max_gap_size = max_gap_size
        self.method = method
    
    def fill_gaps(self, data: List[float], fill_method: str = 'interpolate') -> List[float]:
        """Fill gaps in data"""
        if len(data) < 3:
            return data
        
        data_array = np.array(data, dtype=float)
        
        # Convert zeros to NaN for consistent handling
        data_array[data_array == 0] = np.nan
        
        if fill_method == 'interpolate':
            return self._interpolate_gaps(data_array).tolist()
        elif fill_method == 'forward_fill':
            return self._forward_fill(data_array).tolist()
        elif fill_method == 'backward_fill':
            return self._backward_fill(data_array).tolist()
        elif fill_method == 'mean_fill':
            return self._mean_fill(data_array).tolist()
        else:
            return data
    
    def _interpolate_gaps(self, data: np.ndarray) -> np.ndarray:
        """Interpolate gaps using specified method"""
        nan_mask = np.isnan(data)
        
        if not np.any(nan_mask):
            return data
        
        # Find gap segments
        gap_segments = self._find_gap_segments(nan_mask)
        
        result = data.copy()
        
        for start_idx, end_idx in gap_segments:
            gap_size = end_idx - start_idx
            
            if gap_size <= self.max_gap_size:
                # Interpolate this gap
                result = self._interpolate_segment(result, start_idx, end_idx)
        
        return result
    
    def _find_gap_segments(self, nan_mask: np.ndarray) -> List[tuple]:
        """Find continuous NaN segments"""
        segments = []
        start_idx = None
        
        for i, is_nan in enumerate(nan_mask):
            if is_nan and start_idx is None:
                start_idx = i
            elif not is_nan and start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
        
        # Handle case where data ends with NaN
        if start_idx is not None:
            segments.append((start_idx, len(nan_mask)))
        
        return segments
    
    def _interpolate_segment(self, data: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """Interpolate a single gap segment"""
        # Find valid points before and after gap
        before_idx = self._find_last_valid_before(data, start_idx)
        after_idx = self._find_first_valid_after(data, end_idx)
        
        if before_idx is not None and after_idx is not None:
            # Linear interpolation
            x_points = [before_idx, after_idx]
            y_points = [data[before_idx], data[after_idx]]
            x_interp = list(range(start_idx, end_idx))
            
            if self.method == 'linear':
                y_interp = np.interp(x_interp, x_points, y_points)
            elif self.method == 'cubic' and len(x_points) >= 4:
                # Need more points for cubic
                extended_before = max(0, before_idx - 2)
                extended_after = min(len(data), after_idx + 2)
                
                x_extended = list(range(extended_before, extended_after))
                y_extended = [data[i] for i in x_extended if not np.isnan(data[i])]
                x_extended = [i for i in x_extended if not np.isnan(data[i])]
                
                if len(x_extended) >= 4:
                    f = interpolate.interp1d(x_extended, y_extended, kind='cubic', 
                                           bounds_error=False, fill_value='extrapolate')
                    y_interp = f(x_interp)
                else:
                    y_interp = np.interp(x_interp, x_points, y_points)
            else:
                y_interp = np.interp(x_interp, x_points, y_points)
            
            data[start_idx:end_idx] = y_interp
            
        elif before_idx is not None:
            # Forward fill
            data[start_idx:end_idx] = data[before_idx]
        elif after_idx is not None:
            # Backward fill
            data[start_idx:end_idx] = data[after_idx]
        
        return data
    
    def _find_last_valid_before(self, data: np.ndarray, index: int) -> Optional[int]:
        """Find last valid data point before given index"""
        for i in range(index - 1, -1, -1):
            if not np.isnan(data[i]):
                return i
        return None
    
    def _find_first_valid_after(self, data: np.ndarray, index: int) -> Optional[int]:
        """Find first valid data point after given index"""
        for i in range(index, len(data)):
            if not np.isnan(data[i]):
                return i
        return None
    
    def _forward_fill(self, data: np.ndarray) -> np.ndarray:
        """Forward fill NaN values"""
        result = data.copy()
        last_valid = None
        
        for i in range(len(result)):
            if not np.isnan(result[i]):
                last_valid = result[i]
            elif last_valid is not None:
                result[i] = last_valid
        
        return result
    
    def _backward_fill(self, data: np.ndarray) -> np.ndarray:
        """Backward fill NaN values"""
        result = data.copy()
        next_valid = None
        
        for i in range(len(result) - 1, -1, -1):
            if not np.isnan(result[i]):
                next_valid = result[i]
            elif next_valid is not None:
                result[i] = next_valid
        
        return result
    
    def _mean_fill(self, data: np.ndarray) -> np.ndarray:
        """Fill NaN values with mean of valid data"""
        result = data.copy()
        valid_data = result[~np.isnan(result)]
        
        if len(valid_data) > 0:
            mean_value = np.mean(valid_data)
            result[np.isnan(result)] = mean_value
        
        return result