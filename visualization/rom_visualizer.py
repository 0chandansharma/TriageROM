#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Tuple, Optional
import io

class ROMVisualizer:
    """Visualize Range of Motion analysis results"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Colors
        self.colors = {
            'normal_range': (0, 255, 0),      # Green
            'achieved_range': (0, 255, 255),  # Yellow
            'deficit_range': (0, 0, 255),     # Red
            'current_position': (255, 255, 255), # White
            'text': (255, 255, 255),          # White
            'background': (50, 50, 50)        # Dark gray
        }
    
    def draw_rom_summary(self, image: np.ndarray, rom_results: Dict, 
                        position: Tuple[int, int] = (20, 20)) -> np.ndarray:
        """Draw ROM analysis summary on image"""
        x, y = position
        
        # Extract data
        final_results = rom_results.get('final_results', {})
        flexion_rom = final_results.get('flexion_rom', 0)
        extension_rom = final_results.get('extension_rom', 0)
        total_rom = final_results.get('total_rom', 0)
        assessment = final_results.get('assessment', 'unknown')
        rom_percentage = final_results.get('rom_percentage', 0)
        
        # Normal ranges
        normal_range = final_results.get('normal_range', {})
        normal_flexion = normal_range.get('flexion', 50)
        normal_extension = normal_range.get('extension', 15)
        normal_total = normal_range.get('total', 65)
        
        # Draw summary box
        box_width = 300
        box_height = 180
        
        # Background
        cv2.rectangle(image, (x, y), (x + box_width, y + box_height), 
                     self.colors['background'], -1)
        cv2.rectangle(image, (x, y), (x + box_width, y + box_height), 
                     self.colors['text'], 2)
        
        # Title
        title = "ROM Analysis Summary"
        cv2.putText(image, title, (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # ROM values
        y_offset = 50
        rom_texts = [
            f"Flexion ROM: {flexion_rom:.1f}° / {normal_flexion:.0f}°",
            f"Extension ROM: {extension_rom:.1f}° / {normal_extension:.0f}°",
            f"Total ROM: {total_rom:.1f}° / {normal_total:.0f}°",
            f"Achievement: {rom_percentage:.1f}%",
            f"Assessment: {assessment.replace('_', ' ').title()}"
        ]
        
        for text in rom_texts:
            cv2.putText(image, text, (x + 10, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 25
        
        return image
    
    def draw_rom_gauge(self, image: np.ndarray, current_angle: float, 
                      max_flexion: float, max_extension: float,
                      normal_flexion: float = 50, normal_extension: float = 15,
                      position: Tuple[int, int] = (400, 50)) -> np.ndarray:
        """Draw ROM gauge showing current position and ranges"""
        x, y = position
        gauge_radius = 80
        center = (x + gauge_radius, y + gauge_radius)
        
        # Calculate angles for gauge (semicircle from -90 to +90 degrees)
        # Map ROM values to gauge angles
        total_normal_range = normal_flexion + normal_extension
        total_max_range = abs(max_flexion) + max_extension
        
        # Gauge angles (-90 to +90 degrees, where -90 is max flexion)
        flexion_gauge_angle = -90  # Start of gauge
        neutral_gauge_angle = -90 + (normal_flexion / total_normal_range) * 180
        extension_gauge_angle = 90  # End of gauge
        
        # Current position angle
        if current_angle < 0:  # Flexion
            angle_ratio = abs(current_angle) / normal_flexion
            current_gauge_angle = neutral_gauge_angle - (angle_ratio * (neutral_gauge_angle - flexion_gauge_angle))
        else:  # Extension
            angle_ratio = current_angle / normal_extension
            current_gauge_angle = neutral_gauge_angle + (angle_ratio * (extension_gauge_angle - neutral_gauge_angle))
        
        # Draw gauge background
        cv2.ellipse(image, center, (gauge_radius, gauge_radius), 0, 
                   -90, 90, (100, 100, 100), 8)
        
        # Draw normal range arc
        cv2.ellipse(image, center, (gauge_radius, gauge_radius), 0, 
                   flexion_gauge_angle, extension_gauge_angle, 
                   self.colors['normal_range'], 6)
        
        # Draw achieved range arc
        achieved_start = -90 + (abs(max_flexion) / total_normal_range) * 180
        achieved_end = -90 + ((abs(max_flexion) + max_extension) / total_normal_range) * 180
        
        cv2.ellipse(image, center, (gauge_radius - 10, gauge_radius - 10), 0, 
                   achieved_start, achieved_end, self.colors['achieved_range'], 4)
        
        # Draw current position indicator
        angle_rad = math.radians(current_gauge_angle)
        indicator_start = (
            int(center[0] + (gauge_radius - 20) * math.cos(angle_rad)),
            int(center[1] + (gauge_radius - 20) * math.sin(angle_rad))
        )
        indicator_end = (
            int(center[0] + (gauge_radius + 10) * math.cos(angle_rad)),
            int(center[1] + (gauge_radius + 10) * math.sin(angle_rad))
        )
        
        cv2.line(image, indicator_start, indicator_end, 
                self.colors['current_position'], 3)
        
        # Draw center point
        cv2.circle(image, center, 5, self.colors['text'], -1)
        
        # Draw labels
        cv2.putText(image, f"Flex: {normal_flexion:.0f}°", 
                   (x - 30, y + gauge_radius * 2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        cv2.putText(image, f"Ext: {normal_extension:.0f}°", 
                   (x + gauge_radius + 20, y + gauge_radius * 2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        cv2.putText(image, f"Current: {current_angle:.1f}°", 
                   (x + 20, y + gauge_radius * 2 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['current_position'], 1)
        
        return image
    
    def create_rom_chart(self, movement_history: List[Dict], 
                        title: str = "ROM Analysis Over Time") -> np.ndarray:
        """Create ROM chart using matplotlib and convert to OpenCV image"""
        if not movement_history:
            # Return empty chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(title)
            return self._fig_to_cv2(fig)
        
        # Extract data
        timestamps = []
        trunk_angles = []
        
        for i, data in enumerate(movement_history):
            timestamps.append(i / 30.0)  # Assume 30 FPS
            if 'loweback_analysis' in data:
                trunk_angles.append(data['loweback_analysis'].get('trunk_angle', 0))
            else:
                trunk_angles.append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot angle over time
        ax.plot(timestamps, trunk_angles, 'b-', linewidth=2, label='Trunk Angle')
        
        # Add reference lines
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Neutral')
        ax.axhline(y=-50, color='r', linestyle='--', alpha=0.5, label='Target Flexion')
        ax.axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Target Extension')
        
        # Fill areas
        ax.fill_between(timestamps, trunk_angles, 0, 
                       where=np.array(trunk_angles) < 0, 
                       alpha=0.3, color='red', label='Flexion')
        ax.fill_between(timestamps, trunk_angles, 0, 
                       where=np.array(trunk_angles) > 0, 
                       alpha=0.3, color='green', label='Extension')
        
        # Formatting
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Trunk Angle (degrees)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if trunk_angles:
            max_flexion = min(trunk_angles)
            max_extension = max(trunk_angles)
            total_rom = max_extension - max_flexion
            
            stats_text = f'Max Flexion: {max_flexion:.1f}°\n'
            stats_text += f'Max Extension: {max_extension:.1f}°\n'
            stats_text += f'Total ROM: {total_rom:.1f}°'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return self._fig_to_cv2(fig)
    
    def _fig_to_cv2(self, fig) -> np.ndarray:
        """Convert matplotlib figure to OpenCV image"""
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get the RGBA buffer from the figure
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        
        # Convert RGB to BGR for OpenCV
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        return buf
    
    def draw_movement_phases(self, image: np.ndarray, movement_history: List[Dict], 
                           position: Tuple[int, int] = (20, 250)) -> np.ndarray:
        """Draw movement phases timeline"""
        if not movement_history:
            return image
        
        x, y = position
        timeline_width = 400
        timeline_height = 30
        
        # Background
        cv2.rectangle(image, (x, y), (x + timeline_width, y + timeline_height), 
                     self.colors['background'], -1)
        
        # Draw phases
        phase_colors = {
            'neutral': (128, 128, 128),
            'flexing': (0, 255, 255),
            'extending': (0, 255, 0),
            'deep_flexion': (0, 100, 255),
            'extension': (0, 200, 0)
        }
        
        segment_width = timeline_width / len(movement_history)
        
        for i, data in enumerate(movement_history):
            if 'loweback_analysis' in data:
                phase = data['loweback_analysis'].get('movement_phase', 'unknown')
                color = phase_colors.get(phase, (100, 100, 100))
                
                segment_x = x + int(i * segment_width)
                cv2.rectangle(image, (segment_x, y), 
                            (segment_x + int(segment_width), y + timeline_height), 
                            color, -1)
        
        # Labels
        cv2.putText(image, "Movement Phases Timeline", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Legend
        legend_y = y + timeline_height + 20
        legend_x = x
        
        for phase, color in phase_colors.items():
            cv2.rectangle(image, (legend_x, legend_y), (legend_x + 15, legend_y + 15), 
                         color, -1)
            cv2.putText(image, phase.title(), (legend_x + 20, legend_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            legend_x += 80
        
        return image