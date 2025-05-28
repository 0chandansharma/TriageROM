#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List

class GaussianFilter:
    """Gaussian filter implementation"""
    
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
    
    def filter_data(self, data: List[float]) -> List[float]:
        """Apply Gaussian filter to data"""
        if len(data) < 3:
            return data
        
        try:
            data_array = np.array(data)
            nan_mask = np.isnan(data_array)
            
            if np.all(nan_mask):
                return data
            
            # Handle NaN values by interpolation or by processing segments
            if np.any(nan_mask):
                # Find valid data points
                valid_indices = np.where(~nan_mask)[0]
                valid_data = data_array[valid_indices]
                
                if len(valid_data) < 3:
                    return data
                
                # Apply filter to valid data
                filtered_valid = gaussian_filter1d(valid_data, sigma=self.sigma)
                
                # Reconstruct full array
                filtered_data = data_array.copy()
                filtered_data[valid_indices] = filtered_valid
                
                return filtered_data.tolist()
            else:
                # No NaN values, filter directly
                filtered_data = gaussian_filter1d(data_array, sigma=self.sigma)
                return filtered_data.tolist()
                
        except Exception as e:
            print(f"Gaussian filtering error: {e}")
            return data
    
    def get_filter_info(self) -> dict:
        """Get filter information"""
        return {
            "type": "gaussian",
            "sigma": self.sigma
        }