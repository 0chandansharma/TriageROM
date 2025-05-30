#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class AngleVisualizer:
    """Visualize angles and measurements on pose"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Colors
        self.colors = {
            'angle_arc': (0, 255, 255),        # Yellow
            'angle_line': (255, 255, 0),       # Cyan
            'angle_text': (255, 255, 255),     # White
            'angle_bg': (0, 0, 0),             # Black
            'reference_line': (128, 128, 128), # Gray
            'good_angle': (0, 255, 0),         # Green
            'warning_angle': (0, 255, 255),    # Yellow
            'bad_angle': (0, 0, 255)           # Red
        }
        
        # Drawing parameters
        self.arc_radius = self.config.get('arc_radius', 40)
        self.line_thickness = self.config.get('line_thickness', 2)
        self.text_scale = self.config.get('text_scale', 0.8)
        
    def draw_trunk_angle(self, image: np.ndarray, spine_keypoints: Dict, 
                        trunk_angle: float) -> np.ndarray:
        """Draw trunk angle visualization"""
        if not self._validate_spine_keypoints(spine_keypoints):
            return image
        
        height, width = image.shape[:2]
        
        # Get keypoint positions
        left_shoulder = spine_keypoints.get('left_shoulder', {})
        right_shoulder = spine_keypoints.get('right_shoulder', {})
        left_hip = spine_keypoints.get('left_hip', {})
        right_hip = spine_keypoints.get('right_hip', {})
        
        # Calculate midpoints
        shoulder_mid = (
            (left_shoulder.get('x', 0) + right_shoulder.get('x', 0)) / 2 * width,
            (left_shoulder.get('y', 0) + right_shoulder.get('y', 0)) / 2 * height
        )
        
        hip_mid = (
            (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2 * width,
            (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2 * height
        )
        
        # Draw trunk line
        cv2.line(image, 
                (int(shoulder_mid[0]), int(shoulder_mid[1])),
                (int(hip_mid[0]), int(hip_mid[1])),
                self.colors['angle_line'], self.line_thickness)
        
        # Draw vertical reference line
        ref_line_start = (int(hip_mid[0]), int(hip_mid[1]))
        ref_line_end = (int(hip_mid[0]), int(hip_mid[1] - 100))
        
        cv2.line(image, ref_line_start, ref_line_end, 
                self.colors['reference_line'], 1, cv2.LINE_DASHED)
        
        # Draw angle arc and text
        self._draw_angle_arc(image, hip_mid, trunk_angle, ref_line_end, 
                           (shoulder_mid[0], shoulder_mid[1]))
        
        # Draw angle value
        angle_color = self._get_angle_color(trunk_angle, 'trunk')
        angle_text = f"Trunk: {trunk_angle:.1f}°"
        
        text_pos = (int(hip_mid[0] + 60), int(hip_mid[1]))
        self._draw_angle_text(image, angle_text, text_pos, angle_color)
        
        return image
    
    def draw_range_indicator(self, image: np.ndarray, current_angle: float, 
                           max_flexion: float, max_extension: float, 
                           position: Tuple[int, int]) -> np.ndarray:
        """Draw range of motion indicator"""
        x, y = position
        indicator_width = 200
        indicator_height = 20
        
        # Draw background
        cv2.rectangle(image, (x, y), (x + indicator_width, y + indicator_height), 
                     (50, 50, 50), -1)
        
        # Calculate positions
        total_range = max_extension - max_flexion
        if total_range > 0:
            # Current position
            current_pos = ((current_angle - max_flexion) / total_range) * indicator_width
            current_pos = max(0, min(indicator_width, current_pos))
            
            # Draw range bar
            cv2.rectangle(image, (x, y), (x + int(current_pos), y + indicator_height), 
                         self.colors['good_angle'], -1)
            
            # Draw current position marker
            marker_x = x + int(current_pos)
            cv2.line(image, (marker_x, y - 5), (marker_x, y + indicator_height + 5), 
                    self.colors['angle_text'], 2)
        
        # Draw labels
        flexion_text = f"{max_flexion:.0f}°"
        extension_text = f"{max_extension:.0f}°"
        current_text = f"{current_angle:.1f}°"
        
        cv2.putText(image, flexion_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['angle_text'], 1)
        cv2.putText(image, extension_text, (x + indicator_width - 30, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['angle_text'], 1)
        cv2.putText(image, current_text, (x + indicator_width//2 - 15, y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['angle_text'], 1)
        
        return image
    
    def draw_movement_phase_indicator(self, image: np.ndarray, phase: str, 
                                    direction: str, position: Tuple[int, int]) -> np.ndarray:
        """Draw movement phase and direction indicator"""
        x, y = position
        
        # Phase indicator
        phase_text = f"Phase: {phase.replace('_', ' ').title()}"
        direction_text = f"Direction: {direction.replace('_', ' ').title()}"
        
        # Choose colors based on phase
        phase_color = {
            'neutral': self.colors['reference_line'],
            'flexing': self.colors['warning_angle'],
            'extending': self.colors['good_angle'],
            'deep_flexion': self.colors['angle_arc'],
            'extension': self.colors['good_angle']
        }.get(phase.lower(), self.colors['angle_text'])
        
        # Draw phase
        self._draw_text_with_background(image, phase_text, (x, y), phase_color)
        
        # Draw direction
        direction_color = {
            'flexing': (0, 255, 255),  # Yellow
            'extending': (0, 255, 0),  # Green  
            'holding': self.colors['reference_line']
        }.get(direction.lower(), self.colors['angle_text'])
        
        self._draw_text_with_background(image, direction_text, (x, y + 30), direction_color)
        
        return image
    
    def draw_quality_metrics(self, image: np.ndarray, quality_metrics: Dict, 
                           position: Tuple[int, int]) -> np.ndarray:
        """Draw movement quality metrics"""
        x, y = position
        y_offset = 0
        
        metrics_to_show = [
            ('Smoothness', quality_metrics.get('movement_smoothness', 0)),
            ('Stability', quality_metrics.get('pose_stability', 0)),
            ('Confidence', quality_metrics.get('confidence_score', 0))
        ]
        
        for metric_name, metric_value in metrics_to_show:
            # Draw metric bar
            bar_width = 100
            bar_height = 15
            
            # Background
            cv2.rectangle(image, (x + 100, y + y_offset), 
                         (x + 100 + bar_width, y + y_offset + bar_height), 
                         (50, 50, 50), -1)
            
            # Value bar
            value_width = int(metric_value * bar_width)
            color = self._get_quality_color(metric_value)
            cv2.rectangle(image, (x + 100, y + y_offset), 
                         (x + 100 + value_width, y + y_offset + bar_height), 
                         color, -1)
            
            # Text
            text = f"{metric_name}: {metric_value:.2f}"
            cv2.putText(image, text, (x, y + y_offset + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['angle_text'], 1)
            
            y_offset += 25
        
        return image
    
    def _draw_angle_arc(self, image: np.ndarray, center: Tuple[float, float], 
                       angle: float, ref_point: Tuple[int, int], 
                       target_point: Tuple[float, float]):
        """Draw angle arc between reference and target"""
        # Calculate angles for arc
        ref_angle = math.degrees(math.atan2(ref_point[1] - center[1], 
                                          ref_point[0] - center[0]))
        target_angle = math.degrees(math.atan2(target_point[1] - center[1], 
                                             target_point[0] - center[0]))
        
        # Normalize angles
        ref_angle = (ref_angle + 360) % 360
        target_angle = (target_angle + 360) % 360
        
        # Draw arc
        start_angle = min(ref_angle, target_angle)
        end_angle = max(ref_angle, target_angle)
        
        # Adjust for crossing 0 degrees
        if abs(end_angle - start_angle) > 180:
            start_angle, end_angle = end_angle, start_angle + 360
        
        cv2.ellipse(image, (int(center[0]), int(center[1])), 
                   (self.arc_radius, self.arc_radius), 0, 
                   start_angle, end_angle, self.colors['angle_arc'], 2)
    
    def _draw_angle_text(self, image: np.ndarray, text: str, 
                        position: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw angle text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, self.text_scale, thickness)
        
        x, y = position
        
        # Background
        cv2.rectangle(image, 
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + baseline + 5),
                     self.colors['angle_bg'], -1)
        
        # Text
        cv2.putText(image, text, position, font, self.text_scale, color, thickness)
    
    def _draw_text_with_background(self, image: np.ndarray, text: str, 
                                 position: Tuple[int, int], 
                                 color: Tuple[int, int, int]):
        """Draw text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness)
        
        x, y = position
        
        # Background
        cv2.rectangle(image, 
                     (x - 3, y - text_height - 3),
                     (x + text_width + 3, y + baseline + 3),
                     self.colors['angle_bg'], -1)
        
        # Text
        cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    def _get_angle_color(self, angle: float, angle_type: str) -> Tuple[int, int, int]:
        """Get color based on angle value and type"""
        if angle_type == 'trunk':
            if -10 <= angle <= 10:  # Neutral range
                return self.colors['good_angle']
            elif -45 <= angle <= 20:  # Acceptable range
                return self.colors['warning_angle']
            else:  # Extreme range
                return self.colors['bad_angle']
        
        return self.colors['angle_text']
    
    def _get_quality_color(self, value: float) -> Tuple[int, int, int]:
        """Get color based on quality metric value"""
        if value >= 0.8:
            return self.colors['good_angle']
        elif value >= 0.6:
            return self.colors['warning_angle']
        else:
            return self.colors['bad_angle']
    
    def _validate_spine_keypoints(self, spine_keypoints: Dict) -> bool:
        """Validate that required spine keypoints are present"""
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        for point in required_points:
            if point not in spine_keypoints:
                return False
            kp = spine_keypoints[point]
            if not isinstance(kp, dict) or kp.get('visibility', 0) < 0.5:
                return False
        
        return Truerom_visualizer.py