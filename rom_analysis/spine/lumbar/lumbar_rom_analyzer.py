#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from base_rom_analyzer import BaseROMAnalyzer
from core.angle_calculation.spine_angles import SpineAngleCalculator

class LumbarROMAnalyzer(BaseROMAnalyzer):
    """Lower back Range of Motion analyzer"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.spine_calculator = SpineAngleCalculator()
        
        # ROM targets from config
        self.target_flexion = config.get('target_flexion_rom', 50.0)
        self.target_extension = config.get('target_extension_rom', 15.0)
        
        # Movement detection parameters
        self.movement_threshold = config.get('min_movement_threshold', 2.0)
        self.completion_threshold = config.get('completion_threshold', 0.8)
        self.timeout_seconds = config.get('movement_timeout_seconds', 30)
        
        # State tracking
        self.current_phase = "neutral"  # neutral, flexing, extending, complete
        self.movement_start_angle = None
        self.peak_flexion = 0.0
        self.peak_extension = 0.0
        self.movement_quality_scores = []
        
    def analyze_movement(self, pose_data: Dict) -> Dict:
        """
        Analyze lower back movement from pose data
        
        Args:
            pose_data: Pose estimation results with spine keypoints
            
        Returns:
            Real-time movement analysis
        """
        if not self.session_active:
            self.start_session()
        
        # Calculate spine angles
        spine_angles = self.spine_calculator.calculate_loweback_angles(
            pose_data.get("spine_keypoints", {})
        )
        
        if not spine_angles.get("calculation_successful", False):
            return self._create_error_response(spine_angles.get("error", "Angle calculation failed"))
        
        # Extract key measurements
        trunk_angle = spine_angles.get("trunk_angle")
        net_spine_angle = spine_angles.get("net_spine_angle", trunk_angle)
        quality_metrics = spine_angles.get("quality_metrics", {})
        
        if trunk_angle is None:
            return self._create_error_response("No valid trunk angle calculated")
        
        # Update movement tracking
        movement_analysis = self._update_movement_tracking(trunk_angle, net_spine_angle, quality_metrics)
        
        # Store movement data
        movement_data = {
            "timestamp": time.time(),
            "trunk_angle": trunk_angle,
            "net_spine_angle": net_spine_angle,
            "phase": self.current_phase,
            "quality_metrics": quality_metrics
        }
        self.movement_history.append(movement_data)
        
        # Create response
        response = {
            "status": "analyzing",
            "timestamp": time.time(),
            "session_active": self.session_active,
            "movement_type": "lumbar_flexion_extension",
            "current_state": self.current_phase,
            
            "spine_keypoints": self._format_spine_keypoints(pose_data.get("spine_keypoints", {})),
            
            "loweback_analysis": {
                "trunk_angle": round(trunk_angle, 2),
                "hip_angle": spine_angles.get("hip_angle"),
                "net_spine_angle": round(net_spine_angle, 2) if net_spine_angle else None,
                "movement_phase": movement_analysis["phase"],
                "direction": movement_analysis["direction"],
                "range_tracking": {
                    "max_flexion": round(self.peak_flexion, 2),
                    "max_extension": round(self.peak_extension, 2),
                    "current_rom": round(self.peak_extension - self.peak_flexion, 2),
                    "target_rom": self.target_flexion + self.target_extension
                },
                "quality_metrics": {
                    "movement_smoothness": quality_metrics.get("movement_smoothness", 0.0),
                    "compensatory_movement": 1.0 - quality_metrics.get("pose_stability", 0.0),
                    "pose_stability": quality_metrics.get("pose_stability", 0.0),
                    "confidence_score": quality_metrics.get("confidence_score", 0.0)
                }
            },
            
            "real_time_feedback": self._generate_feedback(movement_analysis) if self.config.get('enable_guidance', False) else None
        }
        
        return response
    
    def _update_movement_tracking(self, trunk_angle: float, net_spine_angle: float, quality_metrics: Dict) -> Dict:
        """Update movement phase and tracking"""
        
        # Update peak angles
        if trunk_angle < self.peak_flexion:
            self.peak_flexion = trunk_angle
        if trunk_angle > self.peak_extension:
            self.peak_extension = trunk_angle
        
        # Determine movement direction
        direction = "holding"
        if len(self.movement_history) >= 3:
            recent_angles = [h["trunk_angle"] for h in self.movement_history[-3:]]
            angle_changes = np.diff(recent_angles)
            avg_change = np.mean(angle_changes)
            
            if avg_change > 1.0:
                direction = "extending"
            elif avg_change < -1.0:
                direction = "flexing"
        
        # Determine movement phase
        phase = self._determine_movement_phase(trunk_angle, direction)
        
        # Store quality score
        self.movement_quality_scores.append(quality_metrics.get("confidence_score", 0.0))
        if len(self.movement_quality_scores) > 50:  # Keep only recent scores
            self.movement_quality_scores.pop(0)
        
        return {
            "phase": phase,
            "direction": direction,
            "peak_flexion": self.peak_flexion,
            "peak_extension": self.peak_extension,
            "current_rom": self.peak_extension - self.peak_flexion
        }
    
    def _determine_movement_phase(self, trunk_angle: float, direction: str) -> str:
        """Determine current movement phase"""
        
        # Neutral position
        if abs(trunk_angle) < 5:
            if self.current_phase == "complete":
                return "complete"
            return "neutral"
        
        # Deep flexion
        elif trunk_angle < -30:
            return "deep_flexion"
        
        # Extension
        elif trunk_angle > 10:
            return "extension"
        
        # Mid-range movement
        elif trunk_angle < -10:
            return "mid_flexion"
        
        else:
            return "mid_range"
    
    def _generate_feedback(self, movement_analysis: Dict) -> Dict:
        """Generate real-time feedback for user guidance"""
        
        phase = movement_analysis["phase"]
        direction = movement_analysis["direction"]
        current_rom = movement_analysis["current_rom"]
        target_rom = self.target_flexion + self.target_extension
        
        progress_percentage = min(100, (current_rom / target_rom) * 100)
        
        # Generate instruction based on phase
        if phase == "neutral":
            instruction = "Begin by slowly bending forward"
        elif phase == "mid_flexion" and direction == "flexing":
            instruction = "Continue bending forward slowly"
        elif phase == "deep_flexion":
            instruction = "Good depth, now slowly return to upright"
        elif phase == "extension":
            instruction = "Now bend backward gently"
        elif direction == "extending":
            instruction = "Continue the movement slowly"
        else:
            instruction = "Move slowly and controlled"
        
        # Generate form cues
        form_cues = []
        if movement_analysis.get("peak_flexion", 0) > -20:
            form_cues.append("Try to bend deeper from your spine")
        
        # Quality-based cues
        avg_quality = np.mean(self.movement_quality_scores[-10:]) if len(self.movement_quality_scores) >= 10 else 0.5
        if avg_quality < 0.7:
            form_cues.append("Move more slowly for better tracking")
        
        return {
            "instruction": instruction,
            "progress_percentage": round(progress_percentage, 1),
            "form_cues": form_cues,
            "warning": None
        }
    
    def detect_movement_completion(self) -> bool:
        """Detect if a complete flexion-extension cycle has been performed"""
        
        if len(self.movement_history) < 20:  # Need sufficient data
            return False
        
        # Check if we've achieved minimum ROM
        total_rom = self.peak_extension - self.peak_flexion
        min_required_rom = (self.target_flexion + self.target_extension) * self.completion_threshold
        
        if total_rom < min_required_rom:
            return False
        
        # Check if we've returned to neutral position
        recent_angles = [h["trunk_angle"] for h in self.movement_history[-5:]]
        if all(abs(angle) < 10 for angle in recent_angles):  # Back to neutral
            self.current_phase = "complete"
            return True
        
        # Check for timeout
        session_duration = time.time() - self.session_start_time
        if session_duration > self.timeout_seconds:
            self.current_phase = "complete"
            return True
        
        return False
    
    def get_rom_summary(self) -> Dict:
        """Get comprehensive ROM analysis summary"""
        
        if not self.movement_history:
            return {"error": "No movement data available"}
        
        # Calculate final metrics
        total_rom = self.peak_extension - self.peak_flexion
        flexion_rom = abs(self.peak_flexion)
        extension_rom = self.peak_extension
        
        # Movement quality analysis
        avg_quality = np.mean(self.movement_quality_scores) if self.movement_quality_scores else 0.0
        
        # Movement pattern analysis
        angles = [h["trunk_angle"] for h in self.movement_history]
        movement_velocity = self._calculate_movement_velocity(angles)
        smoothness_score = self._calculate_smoothness(angles)
        
        # Clinical assessment
        assessment = self._generate_clinical_assessment(flexion_rom, extension_rom, total_rom)
        
        return {
            "status": "completed",
            "session_duration": time.time() - self.session_start_time,
            "movement_type": "flexion_extension_cycle",
            
            "final_results": {
                "flexion_rom": round(flexion_rom, 2),
                "extension_rom": round(extension_rom, 2),
                "total_rom": round(total_rom, 2),
                "normal_range": {
                    "flexion": self.target_flexion,
                    "extension": self.target_extension,
                    "total": self.target_flexion + self.target_extension
                },
                "rom_percentage": round((total_rom / (self.target_flexion + self.target_extension)) * 100, 1),
                "assessment": assessment["category"]
            },
            
            "movement_quality": {
                "overall_score": round(avg_quality * 100, 1),
                "smoothness_score": round(smoothness_score * 100, 1),
                "consistency_score": round(self._calculate_consistency(), 1)
            },
            
            "detailed_metrics": {
                "movement_duration": round(time.time() - self.session_start_time, 1),
                "peak_velocity": round(movement_velocity, 1),
                "data_points": len(self.movement_history),
                "repetitions_completed": self._count_repetitions()
            },
            
            "clinical_insights": assessment["insights"]
        }
    
    def _calculate_movement_velocity(self, angles: List[float]) -> float:
        """Calculate peak movement velocity"""
        if len(angles) < 2:
            return 0.0
        
        velocities = np.abs(np.diff(angles))
        return float(np.max(velocities))
    
    def _calculate_smoothness(self, angles: List[float]) -> float:
        """Calculate movement smoothness (0-1, higher is smoother)"""
        if len(angles) < 3:
            return 0.5
        
        velocities = np.diff(angles)
        accelerations = np.diff(velocities)
        
        if len(accelerations) > 0:
            # Smoothness inversely related to acceleration variance
            smoothness = 1.0 / (1.0 + np.var(accelerations) / 10.0)
            return min(1.0, smoothness)
        
        return 0.5
    
    def _calculate_consistency(self) -> float:
        """Calculate movement consistency score"""
        if len(self.movement_quality_scores) < 5:
            return 50.0
        
        # Consistency based on quality score variance
        variance = np.var(self.movement_quality_scores)
        consistency = max(0, 100 - (variance * 200))  # Scale appropriately
        return consistency
    
    def _count_repetitions(self) -> int:
        """Count number of complete movement repetitions"""
        if len(self.movement_history) < 10:
            return 0
        
        angles = [h["trunk_angle"] for h in self.movement_history]
        
        # Find peaks (local maxima and minima)
        from scipy.signal import find_peaks
        
        # Find flexion peaks (negative values, so invert)
        flexion_peaks, _ = find_peaks([-a for a in angles], height=10, distance=10)
        
        # Find extension peaks (positive values)
        extension_peaks, _ = find_peaks(angles, height=5, distance=10)
        
        # A repetition requires both flexion and extension
        repetitions = min(len(flexion_peaks), len(extension_peaks))
        return max(1, repetitions)  # At least 1 if we have data
    
    def _generate_clinical_assessment(self, flexion_rom: float, extension_rom: float, total_rom: float) -> Dict:
        """Generate clinical assessment of ROM"""
        
        # Compare to normal ranges
        flexion_percent = (flexion_rom / self.target_flexion) * 100
        extension_percent = (extension_rom / self.target_extension) * 100
        total_percent = (total_rom / (self.target_flexion + self.target_extension)) * 100
        
        # Determine overall category
        if total_percent >= 90:
            category = "normal_range"
        elif total_percent >= 75:
            category = "slightly_limited"
        elif total_percent >= 50:
            category = "moderately_limited"
        else:
            category = "severely_limited"
        
        # Generate specific insights
        insights = {
            "primary_limitation": None,
            "recommendations": [],
            "strengths": []
        }
        
        if flexion_percent < 75:
            insights["primary_limitation"] = "flexion_range"
            insights["recommendations"].append("Focus on forward bending exercises")
            insights["recommendations"].append("Improve hip flexor flexibility")
        
        if extension_percent < 75:
            if insights["primary_limitation"]:
                insights["primary_limitation"] = "both_directions"
            else:
                insights["primary_limitation"] = "extension_range"
            insights["recommendations"].append("Strengthen back extensors")
            insights["recommendations"].append("Improve spinal extension mobility")
        
        if flexion_percent >= 90:
            insights["strengths"].append("Good flexion range")
        
        if extension_percent >= 90:
            insights["strengths"].append("Good extension range")
        
        return {
            "category": category,
            "insights": insights,
            "percentages": {
                "flexion": round(flexion_percent, 1),
                "extension": round(extension_percent, 1),
                "total": round(total_percent, 1)
            }
        }
    
    def _format_spine_keypoints(self, keypoints: Dict) -> Dict:
        """Format spine keypoints for API response"""
        required_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        formatted = {
            "required_points": required_points,
            "landmarks": []
        }
        
        for point_name in required_points:
            if point_name in keypoints:
                kp = keypoints[point_name]
                formatted["landmarks"].append({
                    "name": point_name,
                    "x": round(kp.get("x", 0), 4),
                    "y": round(kp.get("y", 0), 4),
                    "z": round(kp.get("z", 0), 4),
                    "visibility": round(kp.get("visibility", 0), 3),
                    "world_x": round(kp.get("world_x", 0), 4),
                    "world_y": round(kp.get("world_y", 0), 4),
                    "world_z": round(kp.get("world_z", 0), 4)
                })
        
        return formatted
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            "status": "error",
            "error_type": "ANALYSIS_ERROR",
            "message": error_message,
            "timestamp": time.time(),
            "session_active": self.session_active
        }
    
    def reset_analysis(self):
        """Reset analysis state"""
        super().reset_analysis()
        self.spine_calculator.reset_history()
        self.current_phase = "neutral"
        self.movement_start_angle = None
        self.peak_flexion = 0.0
        self.peak_extension = 0.0
        self.movement_quality_scores = []