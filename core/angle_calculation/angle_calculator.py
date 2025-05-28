#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class AngleCalculator:
    """Core angle calculation utilities adapted from Sports2D"""
    
    @staticmethod
    def calculate_angle_3_points(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculate angle at point p2 formed by points p1-p2-p3
        Adapted from Sports2D angle calculation
        """
        try:
            # Convert to numpy arrays
            a = np.array(p1)
            b = np.array(p2)  # vertex
            c = np.array(p3)
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle using dot product
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            
            # Clamp to avoid numerical errors
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            # Calculate angle in degrees
            angle = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle)
            
            return float(angle_degrees)
            
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def calculate_segment_angle(p1: Tuple[float, float], p2: Tuple[float, float], reference_horizontal: bool = True) -> float:
        """
        Calculate angle of segment p1-p2 relative to horizontal or vertical
        Adapted from Sports2D segment angle calculation
        """
        try:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx == 0 and dy == 0:
                return np.nan
            
            # Calculate angle relative to horizontal
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            if reference_horizontal:
                # Angle relative to horizontal (0° = horizontal right)
                return float(angle_deg)
            else:
                # Angle relative to vertical (0° = vertical up)
                return float(angle_deg - 90.0)
                
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def calculate_trunk_angle(shoulder_mid: Tuple[float, float], hip_mid: Tuple[float, float]) -> float:
        """
        Calculate trunk angle relative to vertical
        Positive = extension (backward lean)
        Negative = flexion (forward lean)
        """
        try:
            # Calculate vector from hip to shoulder
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]  # Note: y increases downward in image coordinates
            
            # Calculate angle relative to vertical (upward)
            # In image coordinates, vertical up is negative y direction
            angle_rad = math.atan2(dx, -dy)  # Negative dy for upward vertical reference
            angle_deg = math.degrees(angle_rad)
            
            return float(angle_deg)
            
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def calculate_hip_angle(hip: Tuple[float, float], knee: Tuple[float, float], trunk_mid: Tuple[float, float]) -> float:
        """
        Calculate hip flexion angle
        Used to separate hip movement from spine movement
        """
        try:
            return AngleCalculator.calculate_angle_3_points(trunk_mid, hip, knee)
        except:
            return np.nan
    
    @staticmethod
    def vector_magnitude(vector: Tuple[float, float]) -> float:
        """Calculate magnitude of 2D vector"""
        return math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    
    @staticmethod
    def dot_product(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Calculate dot product of two 2D vectors"""
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    @staticmethod
    def smooth_angle_sequence(angles: List[float], window_size: int = 5) -> List[float]:
        """
        Smooth angle sequence using moving average
        Adapted from Sports2D smoothing
        """
        if len(angles) < window_size:
            return angles
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(angles)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(angles), i + half_window + 1)
            
            window_angles = angles[start_idx:end_idx]
            valid_angles = [a for a in window_angles if not np.isnan(a)]
            
            if valid_angles:
                smoothed.append(np.mean(valid_angles))
            else:
                smoothed.append(np.nan)
        
        return smoothed