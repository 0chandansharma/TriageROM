#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List
import numpy as np

class LumbarExtensionAnalyzer:
    """Specialized analyzer for lumbar extension movement"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_extension = config.get('target_extension_rom', 15.0)
        
    def analyze_extension_movement(self, trunk_angle: float, movement_history: List[Dict]) -> Dict:
        """
        Analyze extension-specific parameters
        
        Args:
            trunk_angle: Current trunk angle
            movement_history: History of movement data
            
        Returns:
            Extension analysis results
        """
        
        # Calculate extension-specific metrics
        extension_depth = max(0, trunk_angle)  # Only positive angles count as extension
        extension_percentage = (extension_depth / self.target_extension) * 100
        
        # Analyze extension pattern
        pattern_analysis = self._analyze_extension_pattern(movement_history)
        
        return {
            "extension_depth": round(extension_depth, 2),
            "extension_percentage": round(extension_percentage, 1),
            "target_extension": self.target_extension,
            "pattern_analysis": pattern_analysis,
            "extension_quality": self._assess_extension_quality(extension_depth, pattern_analysis)
        }
    
    def _analyze_extension_pattern(self, movement_history: List[Dict]) -> Dict:
        """Analyze the pattern of extension movement"""
        
        if len(movement_history) < 5:
            return {"insufficient_data": True}
        
        # Extract angles from recent history
        recent_angles = [h["trunk_angle"] for h in movement_history[-10:]]
        extension_angles = [a for a in recent_angles if a > 0]
        
        if not extension_angles:
            return {"no_extension_detected": True}
        
        # Calculate extension rate
        extension_rate = abs(np.mean(np.diff(extension_angles))) if len(extension_angles) > 1 else 0
        
        # Check for compensation patterns
        compensation_detected = any(angle > self.target_extension * 1.5 for angle in extension_angles)
        
        return {
            "extension_rate": round(extension_rate, 2),
            "compensation_detected": compensation_detected,
            "max_extension_achieved": round(max(extension_angles), 2),
            "controlled_movement": extension_rate < 3.0  # Not too fast
        }
    
    def _assess_extension_quality(self, extension_depth: float, pattern_analysis: Dict) -> Dict:
        """Assess the quality of extension movement"""
        
        # Depth score
        depth_score = min(100, (extension_depth / self.target_extension) * 100)
        
        # Control score
        control_score = 90 if pattern_analysis.get("controlled_movement", False) else 60
        
        # Compensation penalty
        compensation_penalty = 20 if pattern_analysis.get("compensation_detected", False) else 0
        
        # Overall quality
        overall_score = max(0, (depth_score + control_score) / 2 - compensation_penalty)
        
        return {
            "depth_score": round(depth_score, 1),
            "control_score": control_score,
            "compensation_penalty": compensation_penalty,
            "overall_score": round(overall_score, 1),
            "quality_level": "excellent" if overall_score >= 85 else 
                           "good" if overall_score >= 70 else 
                           "fair" if overall_score >= 50 else "poor"
        }