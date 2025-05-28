#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import matplotlib.cm as cm

class Sports2DVisualizer:
    """Sports2D style visualization for pose and angles"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.colors = [
            (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
            (0, 255, 255), (0, 0, 0), (255, 255, 255), (125, 0, 0), 
            (0, 125, 0), (0, 0, 125), (125, 125, 0), (125, 0, 125), 
            (0, 125, 125), (255, 125, 125), (125, 255, 125), (125, 125, 255)
        ]
        self.thickness = 2
        
    def draw_pose_sports2d_style(self, image: np.ndarray, pose_results: Dict, 
                                person_id: int = 0, draw_angles: bool = True) -> np.ndarray:
        """Draw complete pose with Sports2D styling"""
        if not pose_results.get("pose_detected", False):
            return image
        
        # Get keypoints and scores
        spine_keypoints = pose_results.get("spine_keypoints", {})
        all_keypoints = pose_results.get("all_pose_landmarks", {}) or spine_keypoints
        
        if not all_keypoints:
            return image
        
        # Convert to lists for drawing
        keypoints_list = []
        scores_list = []
        
        keypoint_names = list(all_keypoints.keys())
        for name in keypoint_names:
            kpt_data = all_keypoints[name]
            if isinstance(kpt_data, dict):
                x = kpt_data.get('pixel_x', kpt_data.get('x', 0) * image.shape[1])
                y = kpt_data.get('pixel_y', kpt_data.get('y', 0) * image.shape[0])
                keypoints_list.append([float(x), float(y)])
                scores_list.append(kpt_data.get('visibility', 0.5))
            else:
                keypoints_list.append([0.0, 0.0])
                scores_list.append(0.0)
        
        # Draw bounding box
        color = self.colors[person_id % len(self.colors)]
        image = self.draw_bounding_box(image, keypoints_list, person_id, color)
        
        # Draw skeleton connections
        image = self.draw_skeleton_connections(image, keypoints_list, scores_list, keypoint_names)
        
        # Draw keypoints
        image = self.draw_keypoints_with_confidence(image, keypoints_list, scores_list)
        
        # Draw angles if enabled
        if draw_angles and "loweback_analysis" in pose_results:
            image = self.draw_angles_sports2d_style(image, pose_results, keypoints_list, keypoint_names)
        
        return image
    
    def draw_keypoints_with_confidence(self, image: np.ndarray, keypoints: List[List[float]], 
                                     scores: List[float], cmap_str: str = 'RdYlGn') -> np.ndarray:
        """Draw keypoints with confidence-based coloring"""
        if not keypoints or not scores:
            return image
        
        cmap = cm.get_cmap(cmap_str)
        
        for kpt, score in zip(keypoints, scores):
            if len(kpt) >= 2 and not np.isnan(kpt[0]) and not np.isnan(kpt[1]):
                x, y = int(kpt[0]), int(kpt[1])
                
                # Skip if point is outside image
                if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                    continue
                
                # Color based on confidence
                normalized_score = max(0, min(1, score))
                color_rgba = cmap(normalized_score)
                color_bgr = (int(color_rgba[2] * 255), int(color_rgba[1] * 255), int(color_rgba[0] * 255))
                
                # Radius based on confidence
                radius = max(2, int(3 + score * 4))
                
                # Draw keypoint
                cv2.circle(image, (x, y), radius, color_bgr, -1)
                cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1)
        
        return image
    
    def draw_skeleton_connections(self, image: np.ndarray, keypoints: List[List[float]], 
                                scores: List[float], keypoint_names: List[str]) -> np.ndarray:
        """Draw skeleton connections based on MediaPipe pose model"""
        # MediaPipe pose connections (simplified)
        connections = [
            # Face
            ('nose', 'left_eye'), ('nose', 'right_eye'),
            ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
            
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Arms
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            
            # Legs
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ]
        
        for connection in connections:
            name1, name2 = connection
            
            if name1 in keypoint_names and name2 in keypoint_names:
                idx1 = keypoint_names.index(name1)
                idx2 = keypoint_names.index(name2)
                
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    idx1 < len(scores) and idx2 < len(scores)):
                    
                    kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
                    score1, score2 = scores[idx1], scores[idx2]
                    
                    if (len(kpt1) >= 2 and len(kpt2) >= 2 and
                        not np.isnan(kpt1[0]) and not np.isnan(kpt1[1]) and
                        not np.isnan(kpt2[0]) and not np.isnan(kpt2[1]) and
                        score1 > 0.3 and score2 > 0.3):
                        
                        pt1 = (int(kpt1[0]), int(kpt1[1]))
                        pt2 = (int(kpt2[0]), int(kpt2[1]))
                        
                        # Skip if points are outside image
                        if (pt1[0] < 0 or pt1[1] < 0 or pt1[0] >= image.shape[1] or pt1[1] >= image.shape[0] or
                            pt2[0] < 0 or pt2[1] < 0 or pt2[0] >= image.shape[1] or pt2[1] >= image.shape[0]):
                            continue
                        
                        # Line color and thickness based on confidence
                        avg_score = (score1 + score2) / 2
                        thickness = max(1, int(avg_score * 3))
                        
                        # Color based on body part
                        if 'shoulder' in name1 or 'shoulder' in name2 or 'hip' in name1 or 'hip' in name2:
                            color = (0, 255, 0)  # Green for torso
                        elif 'elbow' in name1 or 'elbow' in name2 or 'wrist' in name1 or 'wrist' in name2:
                            color = (255, 0, 0)  # Blue for arms
                        elif 'knee' in name1 or 'knee' in name2 or 'ankle' in name1 or 'ankle' in name2:
                            color = (0, 0, 255)  # Red for legs
                        else:
                            color = (255, 255, 255)  # White for face/other
                        
                        cv2.line(image, pt1, pt2, color, thickness)
        
        return image
    
    def draw_bounding_box(self, image: np.ndarray, keypoints: List[List[float]], 
                         person_id: int, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw bounding box around person"""
        valid_points = []
        for kpt in keypoints:
            if len(kpt) >= 2 and not np.isnan(kpt[0]) and not np.isnan(kpt[1]):
                valid_points.append((kpt[0], kpt[1]))
        
        if len(valid_points) < 2:
            return image
        
        x_coords = [pt[0] for pt in valid_points]
        y_coords = [pt[1] for pt in valid_points]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 15
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Person ID label with background
        label = f'Person {person_id}'
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background rectangle for text
        cv2.rectangle(image, (x_min, y_min - text_height - 10), 
                     (x_min + text_width + 10, y_min), color, -1)
        
        # Text
        cv2.putText(image, label, (x_min + 5, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def draw_angles_sports2d_style(self, image: np.ndarray, pose_results: Dict, 
                                  keypoints: List[List[float]], keypoint_names: List[str]) -> np.ndarray:
        """Draw angles in Sports2D style"""
        loweback_analysis = pose_results.get("loweback_analysis", {})
        
        if not loweback_analysis:
            return image
        
        # Get trunk angle
        trunk_angle = loweback_analysis.get("trunk_angle")
        if trunk_angle is None:
            return image
        
        # Find relevant keypoints
        shoulder_points = []
        hip_points = []
        
        for i, name in enumerate(keypoint_names):
            if 'shoulder' in name.lower() and i < len(keypoints):
                kpt = keypoints[i]
                if len(kpt) >= 2 and not np.isnan(kpt[0]) and not np.isnan(kpt[1]):
                    shoulder_points.append(kpt)
            elif 'hip' in name.lower() and i < len(keypoints):
                kpt = keypoints[i]
                if len(kpt) >= 2 and not np.isnan(kpt[0]) and not np.isnan(kpt[1]):
                    hip_points.append(kpt)
        
        if len(shoulder_points) >= 2 and len(hip_points) >= 2:
            # Calculate midpoints
            shoulder_mid = [(shoulder_points[0][0] + shoulder_points[1][0]) / 2,
                           (shoulder_points[0][1] + shoulder_points[1][1]) / 2]
            hip_mid = [(hip_points[0][0] + hip_points[1][0]) / 2,
                      (hip_points[0][1] + hip_points[1][1]) / 2]
            
            # Draw trunk line
            pt1 = (int(hip_mid[0]), int(hip_mid[1]))
            pt2 = (int(shoulder_mid[0]), int(shoulder_mid[1]))
            
            cv2.line(image, pt1, pt2, (0, 255, 0), 3)
            
            # Draw vertical reference line
            ref_length = 100
            vertical_pt = (int(hip_mid[0]), int(hip_mid[1] - ref_length))
            cv2.line(image, pt1, vertical_pt, (255, 255, 0), 2)
            
            # Draw angle arc
            self.draw_angle_arc(image, pt1, pt2, vertical_pt, trunk_angle)
            
            # Draw angle text
            text_pos = (int(hip_mid[0] + 30), int(hip_mid[1] - 30))
            
            # Background for text
            label = f'Trunk: {trunk_angle:.1f}°'
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(image, (text_pos[0] - 5, text_pos[1] - text_height - 5),
                         (text_pos[0] + text_width + 5, text_pos[1] + 5), (0, 0, 0), -1)
            
            cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def draw_angle_arc(self, image: np.ndarray, center_pt: Tuple[int, int], 
                      line1_pt: Tuple[int, int], line2_pt: Tuple[int, int], angle_deg: float):
        """Draw angle arc between two lines"""
        center_x, center_y = center_pt
        
        # Calculate angles for the arc
        angle1 = math.atan2(line1_pt[1] - center_y, line1_pt[0] - center_x)
        angle2 = math.atan2(line2_pt[1] - center_y, line2_pt[0] - center_x)
        
        # Convert to degrees
        start_angle = math.degrees(angle2)
        end_angle = math.degrees(angle1)
        
        # Draw arc
        radius = 30
        cv2.ellipse(image, center_pt, (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 2)