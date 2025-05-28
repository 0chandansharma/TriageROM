#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

class BasePoseEstimator(ABC):
    """Base class for all pose estimation models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the pose estimation model"""
        pass
    
    @abstractmethod
    def estimate_pose(self, image: np.ndarray) -> Dict:
        """
        Estimate pose from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing pose landmarks and metadata
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def validate_keypoints(self, keypoints: Dict, required_keypoints: List[str]) -> bool:
        """Validate that required keypoints are present with sufficient confidence"""
        if not keypoints:
            return False
            
        for kp_name in required_keypoints:
            if kp_name not in keypoints:
                return False
            kp = keypoints[kp_name]
            if kp.get('visibility', 0) < self.config.get('min_visibility', 0.5):
                return False
                
        return True