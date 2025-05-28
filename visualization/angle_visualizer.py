#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import matplotlib.cm as cm

class AngleVisualizer:
    """Comprehensive angle visualization similar to Sports2D"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Angle definitions from Sports2D
        self.angle_dict = {
            # Joint angles
            'right ankle': [['right_knee', 'right_ankle', 'right_big_toe', 'right_heel'], 'dorsiflexion', 90, 1],
            'left ankle': [['left_knee', 'left_ankle', 'left_big_toe', 'left_heel'], 'dorsiflexion', 90, 1],
            'right knee': [['right_ankle', 'right_knee', 'right_hip'], 'flexion', -180, 1],
            'left knee': [['left_ankle', 'left_knee', 'left_hip'], 'flexion', -180, 1],
            'right hip': [['right_knee', 'right_hip', 'hip_center', 'neck'], 'flexion', 0, -1],
            'left hip': [['left_knee', 'left_hip', 'hip_center', 'neck'], 'flexion', 0, -1],
            'right shoulder': [['right_elbow', 'right_shoulder', 'hip_center', 'neck'], 'flexion', 0, -1],
            'left shoulder': [['left_elbow', 'left_shoulder', 'hip_center', 'neck'], 'flexion', 0, -1],
            'right elbow': [['right_wrist', 'right_elbow', 'right_shoulder'], 'flexion', 180, -1],
            'left elbow': [['left_wrist', 'left_elbow', 'left_shoulder'], 'flexion', 180, -1],
            'right wrist': [['right_elbow', 'right_wrist', 'right_index'], 'flexion', -180, 1],
            'left wrist': [['left_elbow', 'left_wrist', 'left_index'], 'flexion', -180, 1],
            
            # Segment angles
            'right foot': [['right_big_toe', 'right_heel'], 'horizontal', 0, -1],
            'left foot': [['left_big_toe', 'left_heel'], 'horizontal', 0, -1],
            'right shank': [['right_ankle', 'right_knee'], 'horizontal', 0, -1],
            'left shank': [['left_ankle', 'left_knee'], 'horizontal', 0, -1],
            'right thigh': [['right_knee', 'right_hip'], 'horizontal', 0, -1],
            'left thigh': [['left_knee', 'left_hip'], 'horizontal', 0, -1],
            'pelvis': [['left_hip', 'right_hip'], 'horizontal', 0, -1],
            'trunk': [['neck', 'hip_center'], 'horizontal', 0, -1],
            'shoulders': [['left_shoulder', 'right_shoulder'], 'horizontal', 0, -1],
            'head': [['head', 'neck'], 'horizontal', 0, -1],
            'right arm': [['right_elbow', 'right_shoulder'], 'horizontal', 0, -1],
            'left arm': [['left_elbow', 'left_shoulder'], 'horizontal', 0, -1],
            'right forearm': [['right_wrist', 'right_elbow'], 'horizontal', 0, -1],
            'left forearm': [['left_wrist', 'left_elbow'], 'horizontal', 0, -1]
        }
        
        # Visualization settings
        self.font_size = config.get('angles', {}).get('fontSize', 0.6)
        self.thickness = config.get('angles', {}).get('thickness', 2)
        self.arc_radius = config.get('visualization', {}).get('angle_arc_radius', 30)
        self.text_offset = config.get('visualization', {}).get('angle_text_offset', 40)
        self.show_reference_lines = config.get('visualization', {}).get('show_reference_lines', True)
    
    def calculate_all_angles(self, keypoints_dict: Dict, keypoint_names: List[str]) -> Dict[str, float]:
        """Calculate all joint and segment angles"""
        
        # Add computed keypoints (neck, hip center)
        enhanced_keypoints = self._add_computed_keypoints(keypoints_dict, keypoint_names)
        
        angles = {}
        
        for angle_name, angle_params in self.angle_dict.items():
            try:
                angle_value = self._calculate_single_angle(angle_name, enhanced_keypoints, angle_params)
                angles[angle_name] = angle_value
            except Exception as e:
                print(f"Error calculating {angle_name}: {e}")
                angles[angle_name] = np.nan
        
        return angles
    
    def _add_computed_keypoints(self, keypoints_dict: Dict, keypoint_names: List[str]) -> Dict:
        """Add computed keypoints like neck and hip center"""
        enhanced = keypoints_dict.copy()
        
        # Add neck (midpoint between shoulders)
        if 'left_shoulder' in keypoints_dict and 'right_shoulder' in keypoints_dict:
            left_shoulder = keypoints_dict['left_shoulder']
            right_shoulder = keypoints_dict['right_shoulder']
            
            if (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                'pixel_x' in left_shoulder and 'pixel_x' in right_shoulder):
                
                enhanced['neck'] = {
                    'pixel_x': (left_shoulder['pixel_x'] + right_shoulder['pixel_x']) // 2,
                    'pixel_y': (left_shoulder['pixel_y'] + right_shoulder['pixel_y']) // 2,
                    'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                    'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
                    'visibility': (left_shoulder['visibility'] + right_shoulder['visibility']) / 2
                }
        
        # Add hip center (midpoint between hips)
        if 'left_hip' in keypoints_dict and 'right_hip' in keypoints_dict:
            left_hip = keypoints_dict['left_hip']
            right_hip = keypoints_dict['right_hip']
            
            if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                'pixel_x' in left_hip and 'pixel_x' in right_hip):
                
                enhanced['hip_center'] = {
                    'pixel_x': (left_hip['pixel_x'] + right_hip['pixel_x']) // 2,
                    'pixel_y': (left_hip['pixel_y'] + right_hip['pixel_y']) // 2,
                    'x': (left_hip['x'] + right_hip['x']) / 2,
                    'y': (left_hip['y'] + right_hip['y']) / 2,
                    'visibility': (left_hip['visibility'] + right_hip['visibility']) / 2
                }
        
        # Add head (above nose if available)
        if 'nose' in keypoints_dict:
            nose = keypoints_dict['nose']
            if isinstance(nose, dict) and 'pixel_x' in nose:
                enhanced['head'] = {
                    'pixel_x': nose['pixel_x'],
                    'pixel_y': max(0, nose['pixel_y'] - 30),  # 30 pixels above nose
                    'x': nose['x'],
                    'y': max(0, nose['y'] - 0.05),  # Approximate head position
                    'visibility': nose['visibility']
                }
        
        return enhanced
    
    def _calculate_single_angle(self, angle_name: str, keypoints_dict: Dict, angle_params: List) -> float:
        """Calculate a single angle based on keypoints"""
        keypoint_names, angle_type, offset, scaling = angle_params
        
        # Get coordinates for required keypoints
        coords = []
        for kp_name in keypoint_names:
            if kp_name in keypoints_dict:
                kp = keypoints_dict[kp_name]
                if isinstance(kp, dict) and 'pixel_x' in kp and 'pixel_y' in kp:
                    if kp.get('visibility', 0) > 0.3:  # Minimum confidence
                        coords.append([kp['pixel_x'], kp['pixel_y']])
                    else:
                        return np.nan
                else:
                    return np.nan
            else:
                # Try alternative names
                alt_name = self._get_alternative_keypoint_name(kp_name)
                if alt_name and alt_name in keypoints_dict:
                    kp = keypoints_dict[alt_name]
                    if isinstance(kp, dict) and 'pixel_x' in kp and 'pixel_y' in kp:
                        if kp.get('visibility', 0) > 0.3:
                            coords.append([kp['pixel_x'], kp['pixel_y']])
                        else:
                            return np.nan
                    else:
                        return np.nan
                else:
                    return np.nan
        
        if len(coords) < 2:
            return np.nan
        
        # Calculate angle based on type
        if angle_type == 'horizontal':
            # Segment angle relative to horizontal
            if len(coords) >= 2:
                return self._calculate_segment_angle(coords[0], coords[1], offset, scaling)
        else:
            # Joint angle
            if len(coords) >= 3:
                return self._calculate_joint_angle(coords, offset, scaling)
        
        return np.nan
    
    def _get_alternative_keypoint_name(self, name: str) -> Optional[str]:
        """Get alternative keypoint names for different models"""
        alternatives = {
            'right_big_toe': 'right_foot_index',
            'left_big_toe': 'left_foot_index',
            'right_index': 'right_wrist',  # Fallback if finger not available
            'left_index': 'left_wrist'
        }
        return alternatives.get(name)
    
    def _calculate_segment_angle(self, p1: List[float], p2: List[float], offset: float, scaling: float) -> float:
        """Calculate segment angle relative to horizontal"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0 and dy == 0:
            return np.nan
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return (angle_deg + offset) * scaling
    
    def _calculate_joint_angle(self, coords: List[List[float]], offset: float, scaling: float) -> float:
        """Calculate joint angle between three points"""
        if len(coords) < 3:
            return np.nan
        
        # Use middle point as vertex
        vertex = coords[1]
        p1 = coords[0]
        p2 = coords[2] if len(coords) > 2 else coords[-1]
        
        # Vectors from vertex
        v1 = [p1[0] - vertex[0], p1[1] - vertex[1]]
        v2 = [p2[0] - vertex[0], p2[1] - vertex[1]]
        
        # Calculate angle
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return np.nan
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return (angle_deg + offset) * scaling
    
    def draw_all_angles_on_image(self, image: np.ndarray, pose_results: Dict, 
                                person_id: int = 0) -> np.ndarray:
        """Draw all calculated angles on the image"""
        
        if not pose_results.get("pose_detected", False):
            return image
        
        # Get keypoints
        all_keypoints = pose_results.get("all_pose_landmarks", {})
        if not all_keypoints:
            all_keypoints = pose_results.get("spine_keypoints", {})
        
        if not all_keypoints:
            return image
        
        # Calculate all angles
        keypoint_names = list(all_keypoints.keys())
        angles = self.calculate_all_angles(all_keypoints, keypoint_names)
        
        # Enhanced keypoints with computed ones
        enhanced_keypoints = self._add_computed_keypoints(all_keypoints, keypoint_names)
        
        # Draw angles on body
        image = self._draw_angles_on_body(image, angles, enhanced_keypoints)
        
        # Draw angle list
        image = self._draw_angle_list(image, angles, person_id)
        
        return image
    
    def _draw_angles_on_body(self, image: np.ndarray, angles: Dict[str, float], 
                           keypoints_dict: Dict) -> np.ndarray:
        """Draw angle values and arcs on the body"""
        
        for angle_name, angle_value in angles.items():
            if np.isnan(angle_value):
                continue
            
            angle_params = self.angle_dict.get(angle_name)
            if not angle_params:
                continue
            
            keypoint_names, angle_type, offset, scaling = angle_params
            
            # Get keypoint coordinates
            coords = []
            for kp_name in keypoint_names:
                if kp_name in keypoints_dict:
                    kp = keypoints_dict[kp_name]
                    if isinstance(kp, dict) and 'pixel_x' in kp:
                        coords.append((kp['pixel_x'], kp['pixel_y']))
            
            if len(coords) < 2:
                continue
            
            # Draw angle visualization
            if angle_type == 'horizontal':
                # Segment angle
                self._draw_segment_angle_visualization(image, coords, angle_value, angle_name)
            else:
                # Joint angle
                if len(coords) >= 3:
                    self._draw_joint_angle_visualization(image, coords, angle_value, angle_name)
    
    def _draw_segment_angle_visualization(self, image: np.ndarray, coords: List[Tuple[int, int]], 
                                        angle_value: float, angle_name: str):
        """Draw segment angle with reference line and arc"""
        if len(coords) < 2:
            return
        
        p1, p2 = coords[0], coords[1]
        
        # Draw segment line
        cv2.line(image, p1, p2, (255, 255, 255), self.thickness)
        
        if self.show_reference_lines:
            # Draw horizontal reference line
            ref_length = 50
            horizontal_end = (p1[0] + ref_length, p1[1])
            cv2.line(image, p1, horizontal_end, (255, 255, 0), 1)
            
            # Draw angle arc
            self._draw_angle_arc(image, p1, p2, horizontal_end, angle_value)
        
        # Draw angle text
        text_pos = (p1[0] + 20, p1[1] - 20)
        self._draw_angle_text(image, f'{angle_value:.1f}°', text_pos, (255, 255, 255))
    
    def _draw_joint_angle_visualization(self, image: np.ndarray, coords: List[Tuple[int, int]], 
                                      angle_value: float, angle_name: str):
        """Draw joint angle with lines and arc"""
        if len(coords) < 3:
            return
        
        vertex = coords[1]
        p1, p2 = coords[0], coords[2]
        
        # Draw lines from vertex
        cv2.line(image, vertex, p1, (0, 255, 0), self.thickness)
        cv2.line(image, vertex, p2, (0, 255, 0), self.thickness)
        
        # Draw angle arc
        self._draw_angle_arc(image, vertex, p1, p2, angle_value)
        
        # Draw angle text
        text_pos = (vertex[0] + 25, vertex[1] - 25)
        self._draw_angle_text(image, f'{angle_value:.1f}°', text_pos, (0, 255, 0))
    
    def _draw_angle_arc(self, image: np.ndarray, center: Tuple[int, int], 
                       p1: Tuple[int, int], p2: Tuple[int, int], angle_value: float):
        """Draw angle arc between two lines"""
        
        # Calculate angles for arc
        angle1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angle2 = math.atan2(p2[1] - center[1], p2[0] - center[0])
        
        start_angle = math.degrees(angle1)
        end_angle = math.degrees(angle2)
        
        # Ensure proper arc direction
        if abs(end_angle - start_angle) > 180:
            if end_angle > start_angle:
                start_angle += 360
            else:
                end_angle += 360
        
        # Draw arc
        try:
            cv2.ellipse(image, center, (self.arc_radius, self.arc_radius), 
                       0, start_angle, end_angle, (0, 255, 255), 2)
        except:
            pass  # Skip if arc parameters are invalid
    
    def _draw_angle_text(self, image: np.ndarray, text: str, position: Tuple[int, int], 
                        color: Tuple[int, int, int]):
        """Draw angle text with background"""
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(image, 
                     (position[0] - 3, position[1] - text_height - 3),
                     (position[0] + text_width + 3, position[1] + 3),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   self.font_size, color, self.thickness)
    
    def _draw_angle_list(self, image: np.ndarray, angles: Dict[str, float], person_id: int):
        """Draw list of all angles on the side of the image"""
        
        # Starting position for angle list
        start_x = 10
        start_y = 30
        line_height = 25
        
        # Draw person header
        person_label = f'Person {person_id} Angles:'
        cv2.putText(image, person_label, (start_x, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        current_y = start_y + 30
        
        # Separate joint and segment angles
        joint_angles = {k: v for k, v in angles.items() 
                       if not np.isnan(v) and any(word in k for word in ['ankle', 'knee', 'hip', 'shoulder', 'elbow', 'wrist'])}
        segment_angles = {k: v for k, v in angles.items() 
                         if not np.isnan(v) and k not in joint_angles}
        
        # Draw joint angles
        if joint_angles:
            cv2.putText(image, 'Joint Angles:', (start_x, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            current_y += 20
            
            for angle_name, angle_value in joint_angles.items():
                text = f'{angle_name}: {angle_value:.1f}°'
                cv2.putText(image, text, (start_x + 10, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                current_y += line_height
        
        # Draw segment angles
        if segment_angles:
            current_y += 10
            cv2.putText(image, 'Segment Angles:', (start_x, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            current_y += 20
            
            for angle_name, angle_value in segment_angles.items():
                text = f'{angle_name}: {angle_value:.1f}°'
                cv2.putText(image, text, (start_x + 10, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                current_y += line_height
        
        return image