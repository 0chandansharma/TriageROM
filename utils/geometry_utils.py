#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import Tuple, List, Optional

class GeometryUtils:
    """Geometric calculation utilities"""
    
    @staticmethod
    def calculate_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two 2D points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def calculate_distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    
    @staticmethod
    def calculate_midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate midpoint between two 2D points"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    @staticmethod
    def calculate_angle_from_horizontal(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate angle of line from horizontal in degrees"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)
    
    @staticmethod
    def calculate_perpendicular_distance(point: Tuple[float, float], 
                                       line_start: Tuple[float, float], 
                                       line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        # Distance formula
        distance = abs(a * x0 + b * y0 + c) / math.sqrt(a**2 + b**2)
        return distance
    
    @staticmethod
    def rotate_point(point: Tuple[float, float], center: Tuple[float, float], angle_deg: float) -> Tuple[float, float]:
        """Rotate point around center by angle in degrees"""
        angle_rad = math.radians(angle_deg)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Translate to origin
        x = point[0] - center[0]
        y = point[1] - center[1]
        
        # Rotate
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        
        # Translate back
        return (x_rot + center[0], y_rot + center[1])
    
    @staticmethod
    def is_point_in_rectangle(point: Tuple[float, float], 
                             rect_top_left: Tuple[float, float],
                             rect_bottom_right: Tuple[float, float]) -> bool:
        """Check if point is inside rectangle"""
        x, y = point
        x1, y1 = rect_top_left
        x2, y2 = rect_bottom_right
        
        return x1 <= x <= x2 and y1 <= y <= y2
    
    @staticmethod
    def calculate_bounding_box(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate bounding box for a set of points"""
        if not points:
            return ((0, 0), (0, 0))
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return ((min_x, min_y), (max_x, max_y))