#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time
from .pose_visualizer import PoseVisualizer
from .angle_visualizer import AngleVisualizer
from .rom_visualizer import ROMVisualizer

class RealTimeDisplay:
    """Real-time display for live pose and ROM analysis"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize visualizers
        self.pose_visualizer = PoseVisualizer(config.get('pose_viz', {}))
        self.angle_visualizer = AngleVisualizer(config.get('angle_viz', {}))
        self.rom_visualizer = ROMVisualizer(config.get('rom_viz', {}))
        
        # Display settings
        self.window_name = self.config.get('window_name', 'TriageROM Live Analysis')
        self.display_width = self.config.get('display_width', 1280)
        self.display_height = self.config.get('display_height', 720)
        
        # Layout settings
        self.main_video_size = (640, 480)
        self.info_panel_width = self.display_width - self.main_video_size[0]
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # Data history for trends
        self.angle_history = []
        self.max_history_length = 300  # 10 seconds at 30fps
        
    def initialize_display(self) -> bool:
        """Initialize display window"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            return True
        except Exception as e:
            print(f"Failed to initialize display: {e}")
            return False
    
    def update_display(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Update real-time display with current frame and analysis"""
        
        # Create main display canvas
        display_canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Process main video frame
        if frame is not None:
            # Resize frame to fit main video area
            resized_frame = cv2.resize(frame, self.main_video_size)
            
            # Add pose visualization if available
            if analysis_result.get('status') == 'analyzing':
                resized_frame = self._add_pose_overlay(resized_frame, analysis_result)
            
            # Place main video in canvas
            display_canvas[0:self.main_video_size[1], 0:self.main_video_size[0]] = resized_frame
        
        # Add info panel
        display_canvas = self._add_info_panel(display_canvas, analysis_result)
        
        # Add performance info
        display_canvas = self._add_performance_info(display_canvas)
        
        # Update angle history
        self._update_angle_history(analysis_result)
        
        # Show display
        cv2.imshow(self.window_name, display_canvas)
        
        # Update FPS counter
        self._update_fps_counter()
        
        return display_canvas
    
    def _add_pose_overlay(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Add pose visualization overlay to frame"""
        
        # Draw pose landmarks
        if 'spine_keypoints' in analysis_result:
            frame = self.pose_visualizer.draw_pose(frame, analysis_result)
        
        # Draw angle measurements
        loweback_data = analysis_result.get('loweback_analysis', {})
        if loweback_data:
            spine_keypoints = analysis_result.get('spine_keypoints', {})
            trunk_angle = loweback_data.get('trunk_angle')
            
            if trunk_angle is not None and 'landmarks' in spine_keypoints:
                # Convert spine keypoints format for visualization
                spine_kp_dict = {}
                for landmark in spine_keypoints['landmarks']:
                    spine_kp_dict[landmark['name']] = {
                        'x': landmark['x'],
                        'y': landmark['y'], 
                        'visibility': landmark['visibility']
                    }
                
                frame = self.angle_visualizer.draw_trunk_angle(frame, spine_kp_dict, trunk_angle)
        
        return frame
    
    def _add_info_panel(self, canvas: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Add information panel to the right side"""
        panel_start_x = self.main_video_size[0]
        panel_width = self.info_panel_width
        
        # Background for info panel
        cv2.rectangle(canvas, (panel_start_x, 0), 
                     (panel_start_x + panel_width, self.display_height), 
                     (40, 40, 40), -1)
        
        # Title
        title = "Live ROM Analysis"
        cv2.putText(canvas, title, (panel_start_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_offset = 60
        
        # Session info
        session_id = analysis_result.get('session_id', 'Unknown')
        session_text = f"Session: {session_id[-8:]}"  # Last 8 chars
        cv2.putText(canvas, session_text, (panel_start_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 30
        
        # Analysis status
        status = analysis_result.get('status', 'Unknown')
        status_text = f"Status: {status.replace('_', ' ').title()}"
        status_color = (0, 255, 0) if status == 'analyzing' else (128, 128, 128)
        cv2.putText(canvas, status_text, (panel_start_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_offset += 40
        
        # Lower back analysis info
        if analysis_result.get('status') == 'analyzing':
            loweback_data = analysis_result.get('loweback_analysis', {})
            
            # Current measurements
            trunk_angle = loweback_data.get('trunk_angle', 0)
            movement_phase = loweback_data.get('movement_phase', 'unknown')
            direction = loweback_data.get('direction', 'unknown')
            
            measurements = [
                f"Trunk Angle: {trunk_angle:.1f}°",
                f"Phase: {movement_phase.replace('_', ' ').title()}",
                f"Direction: {direction.replace('_', ' ').title()}"
            ]
            
            cv2.putText(canvas, "Current Measurements:", (panel_start_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_offset += 25
            
            for measurement in measurements:
                cv2.putText(canvas, measurement, (panel_start_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            y_offset += 20
            
            # Range tracking
            range_data = loweback_data.get('range_tracking', {})
            if range_data:
                cv2.putText(canvas, "Range Tracking:", (panel_start_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                y_offset += 25
                
                range_info = [
                    f"Max Flexion: {range_data.get('max_flexion', 0):.1f}°",
                    f"Max Extension: {range_data.get('max_extension', 0):.1f}°",
                    f"Current ROM: {range_data.get('current_rom', 0):.1f}°",
                    f"Target ROM: {range_data.get('target_rom', 0):.1f}°"
                ]
                
                for info in range_info:
                    cv2.putText(canvas, info, (panel_start_x + 10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20
                
                # ROM progress bar
                current_rom = range_data.get('current_rom', 0)
                target_rom = range_data.get('target_rom', 65)
                progress = min(100, (current_rom / target_rom) * 100) if target_rom > 0 else 0
                
                y_offset += 10
                self._draw_progress_bar(canvas, (panel_start_x + 10, y_offset), 
                                      200, 20, progress, "ROM Progress")
                y_offset += 40
            
            # Quality metrics
            quality_data = loweback_data.get('quality_metrics', {})
            if quality_data:
                cv2.putText(canvas, "Quality Metrics:", (panel_start_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                y_offset += 25
                
                # Quality bars
                metrics = [
                    ('Smoothness', quality_data.get('movement_smoothness', 0)),
                    ('Stability', quality_data.get('pose_stability', 0)),
                    ('Confidence', quality_data.get('confidence_score', 0))
                ]
                
                for metric_name, value in metrics:
                    self._draw_quality_bar(canvas, (panel_start_x + 10, y_offset), 
                                         180, 15, value, metric_name)
                    y_offset += 25
        
        # Angle trend graph
        if len(self.angle_history) > 10:
            y_offset += 20
            self._draw_angle_trend(canvas, (panel_start_x + 10, y_offset), 
                                 panel_width - 20, 100)
        
        return canvas
    
    def _draw_progress_bar(self, canvas: np.ndarray, position: Tuple[int, int], 
                          width: int, height: int, progress: float, label: str):
        """Draw progress bar"""
        x, y = position
        
        # Background
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (60, 60, 60), -1)
        
        # Progress fill
        fill_width = int((progress / 100.0) * width)
        color = (0, 255, 0) if progress >= 80 else (0, 255, 255) if progress >= 50 else (0, 128, 255)
        cv2.rectangle(canvas, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (200, 200, 200), 1)
        
        # Text
        text = f"{label}: {progress:.1f}%"
        cv2.putText(canvas, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_quality_bar(self, canvas: np.ndarray, position: Tuple[int, int], 
                         width: int, height: int, value: float, label: str):
        """Draw quality metric bar"""
        x, y = position
        
        # Background
        cv2.rectangle(canvas, (x + 80, y), (x + 80 + width, y + height), (60, 60, 60), -1)
        
        # Value fill
        fill_width = int(value * width)
        color = (0, 255, 0) if value >= 0.8 else (0, 255, 255) if value >= 0.6 else (0, 0, 255)
        cv2.rectangle(canvas, (x + 80, y), (x + 80 + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(canvas, (x + 80, y), (x + 80 + width, y + height), (200, 200, 200), 1)
        
        # Label and value
        text = f"{label}:"
        cv2.putText(canvas, text, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        value_text = f"{value:.2f}"
        cv2.putText(canvas, value_text, (x + 80 + width + 5, y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_angle_trend(self, canvas: np.ndarray, position: Tuple[int, int], 
                         width: int, height: int):
        """Draw angle trend graph"""
        x, y = position
        
        # Background
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 1)
        
        # Title
        cv2.putText(canvas, "Angle Trend", (x + 5, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(self.angle_history) < 2:
            return
        
        # Find min/max for scaling
        angles = [a for a in self.angle_history if a is not None]
        if not angles:
            return
        
        min_angle = min(angles)
        max_angle = max(angles)
        angle_range = max_angle - min_angle
        
        if angle_range == 0:
            angle_range = 1
        
        # Draw trend line
        graph_area_y = y + 20
        graph_area_height = height - 30
        
        points = []
        for i, angle in enumerate(self.angle_history[-width:]):
            if angle is not None:
                graph_x = x + i
                graph_y = graph_area_y + graph_area_height - int(((angle - min_angle) / angle_range) * graph_area_height)
                points.append((graph_x, graph_y))
        
        # Draw lines between points
        for i in range(1, len(points)):
            cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 1)
        
        # Draw zero line if in range
        if min_angle <= 0 <= max_angle:
            zero_y = graph_area_y + graph_area_height - int(((0 - min_angle) / angle_range) * graph_area_height)
            cv2.line(canvas, (x, zero_y), (x + width, zero_y), (128, 128, 128), 1)
    
    def _add_performance_info(self, canvas: np.ndarray) -> np.ndarray:
        """Add performance information"""
        # FPS display
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(canvas, fps_text, (10, self.display_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Instructions
        instructions = [
            "Controls: Q=Quit, S=New Session, R=Reset, SPACE=Pause"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(canvas, instruction, (10, self.display_height - 50 - (i * 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return canvas
    
    def _update_angle_history(self, analysis_result: Dict):
        """Update angle history for trend display"""
        if analysis_result.get('status') == 'analyzing':
            loweback_data = analysis_result.get('loweback_analysis', {})
            trunk_angle = loweback_data.get('trunk_angle')
            
            if trunk_angle is not None:
                self.angle_history.append(trunk_angle)
            else:
                self.angle_history.append(None)
        else:
            self.angle_history.append(None)
        
        # Maintain history length
        if len(self.angle_history) > self.max_history_length:
            self.angle_history.pop(0)
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def cleanup(self):
        """Cleanup display resources"""
        cv2.destroyWindow(self.window_name)
    
    def handle_key_input(self) -> str:
        """Handle keyboard input and return key pressed"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('s'):
            return 'new_session'
        elif key == ord('r'):
            return 'reset'
        elif key == ord(' '):
            return 'pause'
        elif key == ord('h'):
            return 'help'
        
        return 'none'