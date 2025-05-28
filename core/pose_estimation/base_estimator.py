#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

class BasePoseEstimator(ABC):
    """Base class for all pose estimation models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_initialized = False
        self.skeleton_connections = []
        self.keypoints_colors = []
        self.skeleton_colors = []
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the pose estimation model"""
        pass
    
    @abstractmethod
    def estimate_pose(self, image: np.ndarray) -> Dict:
        """Estimate pose from image"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def get_skeleton_connections(self) -> List[Tuple[int, int]]:
        """Get skeleton connection pairs"""
        return self.skeleton_connections
    
    def get_keypoint_colors(self) -> List[Tuple[int, int, int]]:
        """Get keypoint colors"""
        return self.keypoints_colors
    
    def get_skeleton_colors(self) -> List[Tuple[int, int, int]]:
        """Get skeleton line colors"""
        return self.skeleton_colors
    
    def draw_keypoints_sports2d(self, image: np.ndarray, keypoints: List[List[float]], 
                               scores: List[float], cmap_str: str = 'RdYlGn') -> np.ndarray:
        """Draw keypoints with Sports2D style color mapping"""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        if not keypoints or not scores:
            return image
        
        # Get colormap
        cmap = cm.get_cmap(cmap_str)
        
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if len(kpt) >= 2 and not np.isnan(kpt[0]) and not np.isnan(kpt[1]):
                x, y = int(kpt[0]), int(kpt[1])
                
                # Map score to color (0-1 range)
                normalized_score = max(0, min(1, score))
                color_rgba = cmap(normalized_score)
                color_bgr = (int(color_rgba[2] * 255), int(color_rgba[1] * 255), int(color_rgba[0] * 255))
                
                # Draw circle with size based on confidence
                radius = int(3 + score * 3)  # 3-6 pixels radius
                cv2.circle(image, (x, y), radius, color_bgr, -1)
                cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1)  # Black outline
        
        return image
    
    def draw_skeleton_sports2d(self, image: np.ndarray, keypoints: List[List[float]], 
                              scores: List[float]) -> np.ndarray:
        """Draw skeleton connections with Sports2D style"""
        if not keypoints or not self.skeleton_connections:
            return image
        
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                pt1_idx < len(scores) and pt2_idx < len(scores)):
                
                kpt1, kpt2 = keypoints[pt1_idx], keypoints[pt2_idx]
                score1, score2 = scores[pt1_idx], scores[pt2_idx]
                
                if (len(kpt1) >= 2 and len(kpt2) >= 2 and
                    not np.isnan(kpt1[0]) and not np.isnan(kpt1[1]) and
                    not np.isnan(kpt2[0]) and not np.isnan(kpt2[1]) and
                    score1 > 0.3 and score2 > 0.3):  # Minimum confidence threshold
                    
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    
                    # Line thickness based on average confidence
                    avg_score = (score1 + score2) / 2
                    thickness = max(1, int(avg_score * 3))
                    
                    # Color based on connection type (you can customize this)
                    if any(idx in [11, 12, 23, 24] for idx in connection):  # Torso connections
                        color = (0, 255, 0)  # Green
                    elif any(idx in [13, 14, 15, 16] for idx in connection):  # Arms
                        color = (255, 0, 0)  # Blue
                    elif any(idx in [25, 26, 27, 28] for idx in connection):  # Legs
                        color = (0, 0, 255)  # Red
                    else:
                        color = (255, 255, 255)  # White for other connections
                    
                    cv2.line(image, pt1, pt2, color, thickness)
        
        return image
    
    def draw_bounding_box_sports2d(self, image: np.ndarray, keypoints: List[List[float]], 
                                  person_id: int = 0, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Draw bounding box around person with Sports2D style"""
        if not keypoints:
            return image
        
        # Calculate bounding box
        valid_points = []
        for kpt in keypoints:
            if len(kpt) >= 2 and not np.isnan(kpt[0]) and not np.isnan(kpt[1]):
                valid_points.append((kpt[0], kpt[1]))
        
        if len(valid_points) < 2:
            return image
        
        # Get bounding box coordinates
        x_coords = [pt[0] for pt in valid_points]
        y_coords = [pt[1] for pt in valid_points]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw person ID
        cv2.putText(image, f'Person {person_id}', (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image