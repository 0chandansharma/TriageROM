#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List
import numpy as np

class LumbarFlexionAnalyzer:
    """Specialized analyzer for lumbar flexion movement"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_flexion = config.get('target_flexion_rom', 50.0)
        
    def analyze_flexion_movement(self, trunk_angle: float, movement_history: List[Dict]) -> Dict:
        """
        Analyze flexion-specific parameters
        
        Args:
            trunk_angle: Current trunk angle
            movement_history: History of movement data
            
        Returns:
            Flexion analysis results
        """
        
        # Calculate flexion-specific metrics
        flexion_depth = abs(min(0, trunk_angle))  # Only negative angles count as flexion
        flexion_percentage = (flexion_depth / self.target_flexion) * 100
        
        # Analyze flexion pattern
        pattern_analysis = self._analyze_flexion_pattern(movement_history)
        
        return {
            "flexion_depth": round(flexion_depth, 2),
            "flexion_percentage": round(flexion_percentage, 1),
            "target_flexion": self.target_flexion,
            "pattern_analysis": pattern_analysis,
            "flexion_quality": self._assess_flexion_quality(flexion_depth, pattern_analysis)
        }
    
    def _analyze_flexion_pattern(self, movement_history: List[Dict]) -> Dict:
        """Analyze the pattern of flexion movement"""
        
        if len(movement_history) < 5:
            return {"insufficient_data": True}
        
        # Extract angles from recent history
        recent_angles = [h["trunk_angle"] for h in movement_history[-10:]]
        flexion_angles = [a for a in recent_angles if a < 0]
        
        if not flexion_angles:
            return {"no_flexion_detected": True}
        
        # Calculate flexion rate
        flexion_rate = abs(np.mean(np.diff(flexion_angles))) if len(flexion_angles) > 1 else 0
        
        # Check for hesitation (sudden slowing)
        hesitation_points = []
        if len(flexion_angles) > 3:
            velocities = np.diff(flexion_angles)
            for i, vel in enumerate(velocities):
                if abs(vel) < 0.5 and i > 0:  # Sudden slowing
                    hesitation_points.append(flexion_angles[i])
        
        return {
            "flexion_rate": round(flexion_rate, 2),
            "hesitation_points": hesitation_points,
            "smooth_movement": len(hesitation_points) == 0,
            "max_flexion_achieved": round(min(flexion_angles), 2)
        }
    
    def _assess_flexion_quality(self, flexion_depth: float, pattern_analysis: Dict) -> Dict:
        """Assess the quality of flexion movement"""
        
        # Depth score
        depth_score = min(100, (flexion_depth / self.target_flexion) * 100)
        
        # Smoothness score
        smoothness_score = 90 if pattern_analysis.get("smooth_movement", False) else 60
        
        # Overall quality
        overall_score = (depth_score + smoothness_score) / 2
        
        return {
            "depth_score": round(depth_score, 1),
            "smoothness_score": smoothness_score,
            "overall_score": round(overall_score, 1),
            "quality_level": "excellent" if overall_score >= 85 else 
                           "good" if overall_score >= 70 else 
                           "fair" if overall_score >= 50 else "poor"
        }