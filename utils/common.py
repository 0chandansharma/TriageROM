#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Calculate angle between two 2D vectors in degrees"""
    try:
        # Calculate dot product and magnitudes
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # Calculate angle
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
        
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
        
    except (ValueError, ZeroDivisionError):
        return 0.0

def normalize_vector(vector: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize a 2D vector to unit length"""
    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
    if magnitude == 0:
        return (0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude)

def interpolate_zeros_nans(data: List[float], max_gap: int = 10, method: str = 'linear') -> List[float]:
    """
    Interpolate zeros and NaN values in data sequence
    Adapted from Sports2D interpolation logic
    """
    if not data:
        return data
    
    data_array = np.array(data, dtype=float)
    
    # Convert zeros to NaN for consistent handling
    data_array[data_array == 0] = np.nan
    
    # Find NaN positions
    nan_mask = np.isnan(data_array)
    
    if not np.any(nan_mask):
        return data_array.tolist()
    
    # Find continuous NaN segments
    nan_starts = []
    nan_ends = []
    in_nan_segment = False
    
    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_nan_segment:
            nan_starts.append(i)
            in_nan_segment = True
        elif not is_nan and in_nan_segment:
            nan_ends.append(i - 1)
            in_nan_segment = False
    
    # Handle case where data ends with NaN
    if in_nan_segment:
        nan_ends.append(len(data_array) - 1)
    
    # Interpolate small gaps
    for start, end in zip(nan_starts, nan_ends):
        gap_size = end - start + 1
        
        if gap_size <= max_gap:
            # Find valid data points before and after gap
            before_idx = start - 1 if start > 0 else None
            after_idx = end + 1 if end < len(data_array) - 1 else None
            
            if before_idx is not None and after_idx is not None:
                # Linear interpolation
                before_val = data_array[before_idx]
                after_val = data_array[after_idx]
                
                for i in range(start, end + 1):
                    alpha = (i - before_idx) / (after_idx - before_idx)
                    data_array[i] = before_val + alpha * (after_val - before_val)
            
            elif before_idx is not None:
                # Fill with last valid value
                data_array[start:end+1] = data_array[before_idx]
            
            elif after_idx is not None:
                # Fill with next valid value
                data_array[start:end+1] = data_array[after_idx]
    
    return data_array.tolist()

def compute_height(pose_data: Dict, keypoint_names: List[str], **kwargs) -> float:
    """
    Compute person height from pose data
    Adapted from Sports2D height calculation
    """
    try:
        # Default height if calculation fails
        default_height = kwargs.get('default_height', 1.7)
        
        # Key points for height calculation
        head_points = ['nose', 'head']
        foot_points = ['left_ankle', 'right_ankle', 'left_heel', 'right_heel']
        
        # Find highest and lowest points
        max_y = float('-inf')
        min_y = float('inf')
        
        # Check head points (should be lowest y values in image coordinates)
        for point_name in head_points:
            if point_name in keypoint_names:
                point_data = pose_data.get(point_name)
                if point_data and len(point_data) > 1:  # Assuming [x, y] format
                    y_values = [p[1] if isinstance(p, (list, tuple)) else p for p in point_data if not np.isnan(p)]
                    if y_values:
                        max_y = max(max_y, max(y_values))
        
        # Check foot points (should be highest y values in image coordinates)
        for point_name in foot_points:
            if point_name in keypoint_names:
                point_data = pose_data.get(point_name)
                if point_data and len(point_data) > 1:
                    y_values = [p[1] if isinstance(p, (list, tuple)) else p for p in point_data if not np.isnan(p)]
                    if y_values:
                        min_y = min(min_y, min(y_values))
        
        # Calculate height in pixels
        if max_y > float('-inf') and min_y < float('inf'):
            height_pixels = abs(max_y - min_y)
            return height_pixels if height_pixels > 0 else default_height
        
        return default_height
        
    except Exception as e:
        print(f"Height calculation error: {e}")
        return kwargs.get('default_height', 1.7)

def add_neck_hip_coords(keypoint_name: str, x_coords: np.ndarray, y_coords: np.ndarray, 
                       scores: np.ndarray, keypoint_ids: List[int], keypoint_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add computed neck or hip coordinates when not directly available
    Adapted from Sports2D coordinate addition logic
    """
    try:
        if keypoint_name == 'Neck' and 'Neck' not in keypoint_names:
            # Compute neck as midpoint between shoulders
            if 'left_shoulder' in keypoint_names and 'right_shoulder' in keypoint_names:
                left_idx = keypoint_names.index('left_shoulder')
                right_idx = keypoint_names.index('right_shoulder')
                
                neck_x = (x_coords[left_idx] + x_coords[right_idx]) / 2
                neck_y = (y_coords[left_idx] + y_coords[right_idx]) / 2
                neck_score = (scores[left_idx] + scores[right_idx]) / 2
                
                x_coords = np.append(x_coords, neck_x)
                y_coords = np.append(y_coords, neck_y)
                scores = np.append(scores, neck_score)
        
        elif keypoint_name == 'Hip' and 'Hip' not in keypoint_names:
            # Compute hip center as midpoint between hips
            if 'left_hip' in keypoint_names and 'right_hip' in keypoint_names:
                left_idx = keypoint_names.index('left_hip')
                right_idx = keypoint_names.index('right_hip')
                
                hip_x = (x_coords[left_idx] + x_coords[right_idx]) / 2
                hip_y = (y_coords[left_idx] + y_coords[right_idx]) / 2
                hip_score = (scores[left_idx] + scores[right_idx]) / 2
                
                x_coords = np.append(x_coords, hip_x)
                y_coords = np.append(y_coords, hip_y)
                scores = np.append(scores, hip_score)
        
        return x_coords, y_coords, scores
        
    except Exception as e:
        print(f"Error adding {keypoint_name} coordinates: {e}")
        return x_coords, y_coords, scores

def fixed_angles(coords: List[List[float]], angle_name: str) -> float:
    """
    Calculate angles with fixed conventions
    Adapted from Sports2D angle calculation
    """
    try:
        if len(coords) < 2:
            return np.nan
        
        # For segment angles (2 points)
        if len(coords) == 2:
            p1, p2 = coords
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Calculate angle relative to horizontal
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
        
        # For joint angles (3+ points)
        elif len(coords) >= 3:
            # Use middle point as vertex
            vertex_idx = len(coords) // 2
            vertex = coords[vertex_idx]
            
            # Vectors from vertex to other points
            v1 = [coords[0][0] - vertex[0], coords[0][1] - vertex[1]]
            v2 = [coords[-1][0] - vertex[0], coords[-1][1] - vertex[1]]
            
            return angle_between_vectors(v1, v2)
        
        return np.nan
        
    except Exception as e:
        print(f"Error calculating angle for {angle_name}: {e}")
        return np.nan

def sort_people_sports2d(prev_keypoints: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort people based on Sports2D tracking logic
    Simple distance-based tracking
    """
    try:
        if len(prev_keypoints) == 0 or len(keypoints) == 0:
            return keypoints, keypoints, scores
        
        # Calculate distances between previous and current keypoints
        distances = []
        for i, prev_person in enumerate(prev_keypoints):
            person_distances = []
            for j, curr_person in enumerate(keypoints):
                # Calculate average distance between corresponding keypoints
                valid_distances = []
                for k in range(min(len(prev_person), len(curr_person))):
                    if not (np.isnan(prev_person[k]).any() or np.isnan(curr_person[k]).any()):
                        dist = euclidean_distance(
                            (prev_person[k][0], prev_person[k][1]),
                            (curr_person[k][0], curr_person[k][1])
                        )
                        valid_distances.append(dist)
                
                avg_distance = np.mean(valid_distances) if valid_distances else float('inf')
                person_distances.append(avg_distance)
            
            distances.append(person_distances)
        
        # Hungarian algorithm would be ideal here, but for simplicity use greedy matching
        used_current = set()
        person_mapping = {}
        
        # Sort by distance and assign
        distance_pairs = []
        for i, person_distances in enumerate(distances):
            for j, dist in enumerate(person_distances):
                distance_pairs.append((dist, i, j))
        
        distance_pairs.sort()  # Sort by distance
        
        for dist, prev_idx, curr_idx in distance_pairs:
            if prev_idx not in person_mapping and curr_idx not in used_current:
                person_mapping[prev_idx] = curr_idx
                used_current.add(curr_idx)
        
        # Reorder current keypoints and scores based on mapping
        reordered_keypoints = []
        reordered_scores = []
        
        for i in range(len(prev_keypoints)):
            if i in person_mapping:
                mapped_idx = person_mapping[i]
                reordered_keypoints.append(keypoints[mapped_idx])
                reordered_scores.append(scores[mapped_idx])
            else:
                # Person lost, add empty data
                empty_keypoints = np.full_like(keypoints[0], np.nan)
                empty_scores = np.full_like(scores[0], 0.0)
                reordered_keypoints.append(empty_keypoints)
                reordered_scores.append(empty_scores)
        
        # Add any new people that weren't matched
        for j, curr_person in enumerate(keypoints):
            if j not in used_current:
                reordered_keypoints.append(curr_person)
                reordered_scores.append(scores[j])
        
        return np.array(keypoints), np.array(reordered_keypoints), np.array(reordered_scores)
        
    except Exception as e:
        print(f"Error in people sorting: {e}")
        return keypoints, keypoints, scores