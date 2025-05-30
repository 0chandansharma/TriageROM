#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class PoseVisualizer:
    """Visualize pose landmarks and connections"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.show_all_landmarks = self.config.get('show_all_landmarks', False)
        self.show_spine_only = self.config.get('show_spine_only', True)
        self.show_connections = self.config.get('show_connections', True)
        self.show_confidence = self.config.get('show_confidence', True)
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'spine_landmark': (0, 255, 0),      # Green
            'other_landmark': (255, 0, 0),      # Blue
            'spine_connection': (0, 255, 255),  # Yellow
            'other_connection': (255, 255, 0),  # Cyan
            'low_confidence': (128, 128, 128),  # Gray
            'text': (255, 255, 255),            # White
            'text_bg': (0, 0, 0)                # Black
        }
        
        # Landmark sizes
        self.landmark_radius = self.config.get('landmark_radius', 4)
        self.connection_thickness = self.config.get('connection_thickness', 2)
        
        # Spine connections for visualization
        self.spine_connections = [
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip')
        ]
    
    def draw_pose(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose landmarks and connections on image
        
        Args:
            image: Input image
            pose_data: Pose estimation results
            
        Returns:
            Image with pose visualization
        """
        if not pose_data.get("pose_detected", False):
            return self._draw_no_pose_message(image)
        
        result_image = image.copy()
        
        # Draw spine keypoints
        if "spine_keypoints" in pose_data and self.show_spine_only:
            result_image = self._draw_spine_landmarks(result_image, pose_data["spine_keypoints"])
            
            if self.show_connections:
                result_image = self._draw_spine_connections(result_image, pose_data["spine_keypoints"])
        
        # Draw all pose landmarks if requested
        elif "all_pose_landmarks" in pose_data and self.show_all_landmarks:
            result_image = self._draw_all_landmarks(result_image, pose_data["all_pose_landmarks"])
        
        # Draw tracking info
        if pose_data.get("tracking_active", False):
            result_image = self._draw_tracking_info(result_image, pose_data)
        
        return result_image
    
    def _draw_spine_landmarks(self, image: np.ndarray, spine_keypoints: Dict) -> np.ndarray:
        """Draw spine landmarks"""
        height, width = image.shape[:2]
        
        for kp_name, kp_data in spine_keypoints.items():
            if not isinstance(kp_data, dict):
                continue
            
            x = kp_data.get('x', 0) * width
            y = kp_data.get('y', 0) * height
            visibility = kp_data.get('visibility', 0)
            
            # Choose color based on visibility
            if visibility > 0.7:
                color = self.colors['spine_landmark']
            elif visibility > 0.5:
                color = self.colors['other_landmark']
            else:
                color = self.colors['low_confidence']
            
            # Draw landmark
            cv2.circle(image, (int(x), int(y)), self.landmark_radius, color, -1)
            
            # Draw confidence if enabled
            if self.show_confidence:
                self._draw_confidence_text(image, kp_name, visibility, (int(x), int(y)))
        
        return image
    
    def _draw_all_landmarks(self, image: np.ndarray, all_landmarks: Dict) -> np.ndarray:
        """Draw all pose landmarks"""
        height, width = image.shape[:2]
        spine_keypoints = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        for kp_name, kp_data in all_landmarks.items():
            if not isinstance(kp_data, dict):
                continue
            
            x = kp_data.get('x', 0) * width
            y = kp_data.get('y', 0) * height
            visibility = kp_data.get('visibility', 0)
            
            # Choose color based on keypoint type
            if kp_name in spine_keypoints:
                color = self.colors['spine_landmark']
            else:
                color = self.colors['other_landmark']
            
            # Dim color for low confidence
            if visibility < 0.5:
                color = self.colors['low_confidence']
            
            # Draw landmark
            radius = self.landmark_radius if kp_name in spine_keypoints else self.landmark_radius - 1
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
        
        return image
    
    def _draw_spine_connections(self, image: np.ndarray, spine_keypoints: Dict) -> np.ndarray:
        """Draw connections between spine landmarks"""
        height, width = image.shape[:2]
        
        for connection in self.spine_connections:
            kp1_name, kp2_name = connection
            
            if kp1_name not in spine_keypoints or kp2_name not in spine_keypoints:
                continue
            
            kp1 = spine_keypoints[kp1_name]
            kp2 = spine_keypoints[kp2_name]
            
            if not isinstance(kp1, dict) or not isinstance(kp2, dict):
                continue
            
            # Check visibility
            if (kp1.get('visibility', 0) < 0.5 or kp2.get('visibility', 0) < 0.5):
                continue
            
            # Get coordinates
            x1 = int(kp1.get('x', 0) * width)
            y1 = int(kp1.get('y', 0) * height)
            x2 = int(kp2.get('x', 0) * width)
            y2 = int(kp2.get('y', 0) * height)
            
            # Draw connection
            cv2.line(image, (x1, y1), (x2, y2), 
                    self.colors['spine_connection'], self.connection_thickness)
        
        return image
    
    def _draw_confidence_text(self, image: np.ndarray, kp_name: str, 
                            confidence: float, position: Tuple[int, int]):
        """Draw confidence text near landmark"""
        text = f"{confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position text above landmark
        text_x = position[0] - text_width // 2
        text_y = position[1] - self.landmark_radius - 5
        
        # Draw background
        cv2.rectangle(image, 
                     (text_x - 2, text_y - text_height - 2),
                     (text_x + text_width + 2, text_y + baseline + 2),
                     self.colors['text_bg'], -1)
        
        # Draw text
        cv2.putText(image, text, (text_x, text_y), font, font_scale, 
                   self.colors['text'], thickness)
    
    def _draw_tracking_info(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw tracking information"""
        tracking_info = []
        
        if pose_data.get("tracking_active", False):
            person_id = pose_data.get("person_id", "Unknown")
            tracking_info.append(f"Person ID: {person_id}")
            
            tracking_confidence = pose_data.get("tracking_confidence", 0)
            tracking_info.append(f"Tracking: {tracking_confidence:.2f}")
        
        # Draw tracking info
        y_offset = 30
        for info in tracking_info:
            self._draw_text_with_background(image, info, (10, y_offset))
            y_offset += 25
        
        return image
    
    def _draw_no_pose_message(self, image: np.ndarray) -> np.ndarray:
        """Draw message when no pose is detected"""
        message = "No pose detected"
        height, width = image.shape[:2]
        
        # Position message in center
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)
        
        x = (width - text_width) // 2
        y = (height + text_height) // 2
        
        # Draw with background
        cv2.rectangle(image, 
                     (x - 10, y - text_height - 10),
                     (x + text_width + 10, y + baseline + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(image, message, (x, y), font, font_scale, (0, 0, 255), thickness)
        
        return image
    
    def _draw_text_with_background(self, image: np.ndarray, text: str, 
                                 position: Tuple[int, int], font_scale: float = 0.6):
        """Draw text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        
        # Draw background
        cv2.rectangle(image, 
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + baseline + 5),
                     self.colors['text_bg'], -1)
        
        # Draw text
        cv2.putText(image, text, position, font, font_scale, self.colors['text'], thickness)
    
    def draw_pose_overlay(self, image: np.ndarray, pose_data: Dict, 
                         overlay_info: Dict) -> np.ndarray:
        """Draw pose with additional overlay information"""
        result_image = self.draw_pose(image, pose_data)
        
        # Add overlay information
        if overlay_info:
            y_offset = image.shape[0] - 100  # Start from bottom
            
            for key, value in overlay_info.items():
                text = f"{key}: {value}"
                self._draw_text_with_background(result_image, text, (10, y_offset))
                y_offset += 25
        
        return result_image