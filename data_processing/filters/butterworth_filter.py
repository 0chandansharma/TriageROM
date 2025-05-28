#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from typing import List, Optional

class ButterworthFilter:
    """Butterworth filter implementation"""
    
    def __init__(self, order: int = 4, cutoff_freq: float = 6.0, sampling_rate: float = 30.0):
        self.order = order
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        
        # Calculate filter coefficients
        nyquist_freq = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        if normalized_cutoff >= 1.0:
            raise ValueError(f"Cutoff frequency {cutoff_freq} too high for sampling rate {sampling_rate}")
        
        self.b, self.a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    
    def filter_data(self, data: List[float]) -> List[float]:
        """Apply Butterworth filter to data"""
        if len(data) < 6:  # Need minimum data points
            return data
        
        try:
            # Handle NaN values
            data_array = np.array(data)
            nan_mask = np.isnan(data_array)
            
            if np.all(nan_mask):
                return data
            
            # Find continuous valid segments
            valid_segments = self._find_valid_segments(data_array, nan_mask)
            
            filtered_data = data_array.copy()
            
            # Filter each valid segment
            for start_idx, end_idx in valid_segments:
                segment = data_array[start_idx:end_idx]
                
                if len(segment) >= max(len(self.a), len(self.b)) * 3:
                    # Apply filtfilt for zero-phase filtering
                    filtered_segment = signal.filtfilt(self.b, self.a, segment)
                    filtered_data[start_idx:end_idx] = filtered_segment
            
            return filtered_data.tolist()
            
        except Exception as e:
            print(f"Butterworth filtering error: {e}")
            return data
    
    def _find_valid_segments(self, data: np.ndarray, nan_mask: np.ndarray) -> List[tuple]:
        """Find continuous segments of valid data"""
        segments = []
        start_idx = None
        
        for i, is_nan in enumerate(nan_mask):
            if not is_nan and start_idx is None:
                start_idx = i
            elif is_nan and start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
        
        # Handle case where data ends with valid values
        if start_idx is not None:
            segments.append((start_idx, len(data)))
        
        return segments
    
    def get_filter_info(self) -> dict:
        """Get filter information"""
        return {
            "type": "butterworth",
            "order": self.order,
            "cutoff_frequency": self.cutoff_freq,
            "sampling_rate": self.sampling_rate,
            "normalized_cutoff": self.cutoff_freq / (self.sampling_rate / 2)
        }