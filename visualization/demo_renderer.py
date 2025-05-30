#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from .pose_visualizer import PoseVisualizer
from .angle_visualizer import AngleVisualizer
from .rom_visualizer import ROMVisualizer

class DemoRenderer:
    """Renderer for demonstration mode with enhanced visualizations"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize visualizers
        self.pose_viz = PoseVisualizer(config.get('pose_viz', {}))
        self.angle_viz = AngleVisualizer(config.get('angle_viz', {}))
        self.rom_viz = ROMVisualizer(config.get('rom_viz', {}))
        
        # Demo-specific settings
        self.show_all_overlays = self.config.get('show_all_overlays', True)
        self.show_instructions = self.config.get('show_instructions', True)
        self.show_branding = self.config.get('show_branding', True)
        
        # Colors for demo
        self.demo_colors = {
            'primary': (0, 255, 255),    # Yellow
            'secondary': (255, 0, 255),  # Magenta
            'success': (0, 255, 0),      # Green
            'warning': (0, 255, 255),    # Yellow
            'error': (0, 0, 255),        # Red
            'text': (255, 255, 255),     # White
            'bg': (0, 0, 0)              # Black
        }
    
    def render_demo_frame(self, frame: np.ndarray, analysis_result: Dict, 
                         demo_mode: str = "full") -> np.ndarray:
        """
        Render frame for demonstration purposes
        
        Args:
            frame: Input video frame
            analysis_result: Analysis results
            demo_mode: "full", "minimal", "angles_only", "pose_only"
        """
        if frame is None:
            return self._create_error_frame("No input frame")
        
        demo_frame = frame.copy()
        
        if demo_mode == "full":
            demo_frame = self._render_full_demo(demo_frame, analysis_result)
        elif demo_mode == "minimal":
            demo_frame = self._render_minimal_demo(demo_frame, analysis_result)
        elif demo_mode == "angles_only":
            demo_frame = self._render_angles_only(demo_frame, analysis_result)
        elif demo_mode == "pose_only":
            demo_frame = self._render_pose_only(demo_frame, analysis_result)
        else:
            demo_frame = self._render_full_demo(demo_frame, analysis_result)
        
        # Add demo branding and instructions
        if self.show_branding:
            demo_frame = self._add_branding(demo_frame)
        
        if self.show_instructions:
            demo_frame = self._add_instructions(demo_frame, demo_mode)
        
        return demo_frame
    
    def _render_full_demo(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Render full demonstration with all overlays"""
        
        if analysis_result.get('status') != 'analyzing':
            return self._add_status_overlay(frame, analysis_result)
        
        # Draw pose
        frame = self.pose_viz.draw_pose(frame, analysis_result)
        
        # Draw angles and measurements
        loweback_data = analysis_result.get('loweback_analysis', {})
        if loweback_data:
            # Convert spine keypoints for visualization
            spine_keypoints = self._convert_spine_keypoints(analysis_result)
            
            if spine_keypoints:
                trunk_angle = loweback_data.get('trunk_angle')
                if trunk_angle is not None:
                    frame = self.angle_viz.draw_trunk_angle(frame, spine_keypoints, trunk_angle)
                
                # Movement phase indicator
                phase = loweback_data.get('movement_phase', 'unknown')
                direction = loweback_data.get('direction', 'unknown')
                frame = self.angle_viz.draw_movement_phase_indicator(frame, phase, direction, (10, 80))
                
                # Quality metrics
                quality_metrics = loweback_data.get('quality_metrics', {})
                frame = self.angle_viz.draw_quality_metrics(frame, quality_metrics, (10, 150))
                
                # ROM tracking
                range_data = loweback_data.get('range_tracking', {})
                if range_data:
                    current_angle = trunk_angle or 0
                    max_flexion = range_data.get('max_flexion', 0)
                    max_extension = range_data.get('max_extension', 0)
                    
                    frame = self.angle_viz.draw_range_indicator(frame, current_angle, 
                                                              max_flexion, max_extension, (10, 250))
        
        # Add comprehensive info panel
        frame = self._add_comprehensive_info(frame, analysis_result)
        
        return frame
    
    def _render_minimal_demo(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Render minimal demonstration with key info only"""
        
        if analysis_result.get('status') != 'analyzing':
            return self._add_status_overlay(frame, analysis_result)
        
        # Draw only spine pose
        frame = self.pose_viz.draw_pose(frame, analysis_result)
        
        # Add minimal info
        loweback_data = analysis_result.get('loweback_analysis', {})
        if loweback_data:
            trunk_angle = loweback_data.get('trunk_angle', 0)
            phase = loweback_data.get('movement_phase', 'unknown')
            
            # Simple text overlay
            info_text = [
                f"Angle: {trunk_angle:.1f}°",
                f"Phase: {phase.replace('_', ' ').title()}",
            ]
            
            y_offset = 30
            for text in info_text:
                self._draw_text_with_bg(frame, text, (10, y_offset), 
                                      self.demo_colors['text'], self.demo_colors['bg'])
                y_offset += 30
        
        return frame
    
    def _render_angles_only(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Render with focus on angle measurements"""
        
        if analysis_result.get('status') != 'analyzing':
            return self._add_status_overlay(frame, analysis_result)
        
        loweback_data = analysis_result.get('loweback_analysis', {})
        if loweback_data:
            spine_keypoints = self._convert_spine_keypoints(analysis_result)
            
            if spine_keypoints:
                trunk_angle = loweback_data.get('trunk_angle')
                if trunk_angle is not None:
                    # Draw angle with enhanced visibility
                    frame = self.angle_viz.draw_trunk_angle(frame, spine_keypoints, trunk_angle)
                    
                    # Large angle display
                    angle_text = f"{trunk_angle:.1f}°"
                    cv2.putText(frame, angle_text, (frame.shape[1]//2 - 50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, self.demo_colors['primary'], 3)
                    
                    # Range indicators
                    range_data = loweback_data.get('range_tracking', {})
                    if range_data:
                        max_flexion = range_data.get('max_flexion', 0)
                        max_extension = range_data.get('max_extension', 0)
                        
                        frame = self.angle_viz.draw_range_indicator(frame, trunk_angle, 
                                                                  max_flexion, max_extension, 
                                                                  (frame.shape[1]//2 - 100, 150))
        
        return frame
    
    def _render_pose_only(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Render with focus on pose detection"""
        
        if analysis_result.get('status') != 'analyzing':
            return self._add_status_overlay(frame, analysis_result)
        
        # Enhanced pose visualization
        frame = self.pose_viz.draw_pose(frame, analysis_result)
        
        # Add pose quality info
        if 'spine_keypoints' in analysis_result:
            quality_info = []
            
            # Count visible keypoints
            spine_kp = analysis_result['spine_keypoints']
            if 'landmarks' in spine_kp:
                visible_count = sum(1 for kp in spine_kp['landmarks'] if kp.get('visibility', 0) > 0.5)
                total_count = len(spine_kp['landmarks'])
                quality_info.append(f"Keypoints: {visible_count}/{total_count}")
            
            # Tracking info
            if analysis_result.get('tracking_active'):
                person_id = analysis_result.get('person_id', 'Unknown')
                quality_info.append(f"Tracking ID: {person_id}")
                
                tracking_conf = analysis_result.get('tracking_confidence', 0)
                quality_info.append(f"Confidence: {tracking_conf:.2f}")
            
            # Display quality info
            y_offset = 30
            for info in quality_info:
                self._draw_text_with_bg(frame, info, (10, y_offset), 
                                      self.demo_colors['text'], self.demo_colors['bg'])
                y_offset += 25
        
        return frame
    
    def _add_status_overlay(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Add status overlay for non-analyzing states"""
        status = analysis_result.get('status', 'unknown')
        message = analysis_result.get('message', 'Processing...')
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Status message
        if status == 'no_pose_detected':
            color = self.demo_colors['warning']
            main_text = "No Person Detected"
            sub_text = "Please step into view"
        elif status == 'error':
            color = self.demo_colors['error']
            main_text = "Analysis Error"
            sub_text = message
        else:
            color = self.demo_colors['text']
            main_text = status.replace('_', ' ').title()
            sub_text = message
        
        # Center the text
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        
        cv2.putText(frame, main_text, (center_x - 150, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, sub_text, (center_x - 200, center_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add guidance if no pose detected
        if status == 'no_pose_detected':
            guidance = [
                "• Stand 6-8 feet from camera",
                "• Ensure good lighting",
                "• Show full body in frame",
                "• Face the camera directly"
            ]
            
            y_offset = center_y + 100
            for guide in guidance:
                cv2.putText(frame, guide, (center_x - 180, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_offset += 30
        
        return frame
    
    def _add_comprehensive_info(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Add comprehensive information panel"""
        panel_width = 300
        panel_height = frame.shape[0]
        panel_x = frame.shape[1] - panel_width
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Add border
        cv2.rectangle(frame, (panel_x, 0), (frame.shape[1] - 1, panel_height - 1), 
                     self.demo_colors['primary'], 2)
        
        # Title
        cv2.putText(frame, "Live Analysis", (panel_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.demo_colors['primary'], 2)
        
        y_offset = 60
        
        # Session info
        session_id = analysis_result.get('session_id', 'Unknown')
        session_text = f"Session: ...{session_id[-8:]}" if len(session_id) > 8 else f"Session: {session_id}"
        cv2.putText(frame, session_text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 30
        
        # Lower back analysis
        loweback_data = analysis_result.get('loweback_analysis', {})
        if loweback_data:
            # Current measurements
            cv2.putText(frame, "Measurements:", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.demo_colors['secondary'], 1)
            y_offset += 25
            
            trunk_angle = loweback_data.get('trunk_angle', 0)
            phase = loweback_data.get('movement_phase', 'unknown')
            direction = loweback_data.get('direction', 'unknown')
            
            measurements = [
                f"Trunk: {trunk_angle:.1f}°",
                f"Phase: {phase.replace('_', ' ').title()}",
                f"Dir: {direction.replace('_', ' ').title()}"
            ]
            
            for measurement in measurements:
                cv2.putText(frame, measurement, (panel_x + 15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
            
            y_offset += 10
            
            # Range tracking
            range_data = loweback_data.get('range_tracking', {})
            if range_data:
                cv2.putText(frame, "ROM Tracking:", (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.demo_colors['secondary'], 1)
                y_offset += 25
                
                max_flexion = range_data.get('max_flexion', 0)
                max_extension = range_data.get('max_extension', 0)
                current_rom = range_data.get('current_rom', 0)
                target_rom = range_data.get('target_rom', 65)
                
                rom_info = [
                    f"Flexion: {max_flexion:.1f}°",
                    f"Extension: {max_extension:.1f}°",
                    f"Total: {current_rom:.1f}°",
                    f"Target: {target_rom:.0f}°"
                ]
                
                for info in rom_info:
                    cv2.putText(frame, info, (panel_x + 15, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 20
                
                # Progress bar
                progress = min(100, (current_rom / target_rom) * 100) if target_rom > 0 else 0
                self._draw_mini_progress_bar(frame, (panel_x + 15, y_offset), 200, 15, progress)
                y_offset += 30
            
            # Quality metrics
            quality_data = loweback_data.get('quality_metrics', {})
            if quality_data:
                cv2.putText(frame, "Quality:", (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.demo_colors['secondary'], 1)
                y_offset += 25
                
                metrics = [
                    ('Smooth', quality_data.get('movement_smoothness', 0)),
                    ('Stable', quality_data.get('pose_stability', 0)),
                    ('Conf', quality_data.get('confidence_score', 0))
                ]
                
                for name, value in metrics:
                    color = (0, 255, 0) if value >= 0.8 else (0, 255, 255) if value >= 0.6 else (0, 0, 255)
                    cv2.putText(frame, f"{name}: {value:.2f}", (panel_x + 15, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 20
        
        return frame
    
    def _add_branding(self, frame: np.ndarray) -> np.ndarray:
        """Add TriageROM branding"""
        brand_text = "TriageROM"
        cv2.putText(frame, brand_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.demo_colors['primary'], 2)
        
        version_text = "v1.0 - Live Demo"
        cv2.putText(frame, version_text, (150, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def _add_instructions(self, frame: np.ndarray, demo_mode: str) -> np.ndarray:
        """Add instruction overlay"""
        instructions = {
            "full": ["Q: Quit", "M: Mode", "R: Reset", "Space: Pause"],
            "minimal": ["Q: Quit", "M: Mode", "Minimal view active"],
            "angles_only": ["Q: Quit", "M: Mode", "Angle focus mode"],
            "pose_only": ["Q: Quit", "M: Mode", "Pose detection mode"]
        }
        
        current_instructions = instructions.get(demo_mode, instructions["full"])
        
        # Bottom instruction bar
        bar_height = 30
        bar_y = frame.shape[0] - bar_height
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Instructions
        x_offset = frame.shape[1] - 400
        for instruction in current_instructions:
            cv2.putText(frame, instruction, (x_offset, bar_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            x_offset += 80
        
        return frame
    
    def _convert_spine_keypoints(self, analysis_result: Dict) -> Dict:
        """Convert spine keypoints to format expected by visualizers"""
        spine_keypoints = analysis_result.get('spine_keypoints', {})
        
        if 'landmarks' not in spine_keypoints:
            return {}
        
        converted = {}
        for landmark in spine_keypoints['landmarks']:
            name = landmark.get('name')
            if name:
                converted[name] = {
                    'x': landmark.get('x', 0),
                    'y': landmark.get('y', 0),
                    'visibility': landmark.get('visibility', 0)
                }
        
        return converted
    
    def _draw_text_with_bg(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                          text_color: Tuple[int, int, int], bg_color: Tuple[int, int, int]):
        """Draw text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        
        # Background
        cv2.rectangle(frame, (x - 3, y - text_height - 3), 
                     (x + text_width + 3, y + baseline + 3), bg_color, -1)
        
        # Text
        cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
    
    def _draw_mini_progress_bar(self, frame: np.ndarray, position: Tuple[int, int], 
                               width: int, height: int, progress: float):
        """Draw mini progress bar"""
        x, y = position
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
        
        # Progress
        fill_width = int((progress / 100.0) * width)
        color = (0, 255, 0) if progress >= 80 else (0, 255, 255) if progress >= 50 else (0, 128, 255)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
    def _create_error_frame(self, error_message: str) -> np.ndarray:
        """Create error frame when input is invalid"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.putText(frame, "ERROR", (250, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        cv2.putText(frame, error_message, (150, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame