#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import json
from typing import Dict, List, Optional
from datetime import datetime
import os

class ReportGenerator:
    """Generate comprehensive ROM analysis reports"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def generate_pdf_report(self, session_data: Dict, output_path: str) -> bool:
        """Generate comprehensive PDF report"""
        try:
            with PdfPages(output_path) as pdf:
                # Page 1: Summary
                self._create_summary_page(pdf, session_data)
                
                # Page 2: Detailed Analysis
                self._create_analysis_page(pdf, session_data)
                
                # Page 3: Movement Timeline
                self._create_timeline_page(pdf, session_data)
                
                # Page 4: Clinical Recommendations
                self._create_recommendations_page(pdf, session_data)
            
            return True
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return False
    
    def _create_summary_page(self, pdf: PdfPages, session_data: Dict):
        """Create summary page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('TriageROM Analysis Report - Summary', fontsize=16, fontweight='bold')
        
        # Session info
        session_info = session_data.get('session_info', {})
        final_results = session_data.get('analysis_results', {}).get('final_results', {})
        
        # ROM Summary (top-left)
        ax1.set_title('Range of Motion Summary', fontweight='bold')
        ax1.axis('off')
        
        summary_text = f"""
Session ID: {session_info.get('session_id', 'Unknown')[-12:]}
Date: {datetime.fromisoformat(session_info.get('created_at', '2024-01-01')).strftime('%Y-%m-%d %H:%M')}
Duration: {session_info.get('duration', 0):.1f} seconds

Flexion ROM: {final_results.get('flexion_rom', 0):.1f}°
Extension ROM: {final_results.get('extension_rom', 0):.1f}°
Total ROM: {final_results.get('total_rom', 0):.1f}°

Assessment: {final_results.get('assessment', 'N/A').replace('_', ' ').title()}
ROM Achievement: {final_results.get('rom_percentage', 0):.1f}%
        """
        
        ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', family='monospace')
        
        # ROM Gauge (top-right)
        self._draw_rom_gauge(ax2, final_results)
        
        # Movement Quality (bottom-left)
        movement_quality = session_data.get('analysis_results', {}).get('movement_quality', {})
        self._draw_quality_chart(ax3, movement_quality)
        
        # ROM Comparison (bottom-right)
        self._draw_rom_comparison(ax4, final_results)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_analysis_page(self, pdf: PdfPages, session_data: Dict):
        """Create detailed analysis page"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Detailed Movement Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
        ax1 = fig.add_subplot(gs[0, :])  # Main timeline
        ax2 = fig.add_subplot(gs[1, 0])  # Velocity analysis
        ax3 = fig.add_subplot(gs[1, 1])  # Phase distribution
        ax4 = fig.add_subplot(gs[2, :])  # Statistics table
        
        # Extract movement data
        movement_data = session_data.get('movement_data', [])
        
        if movement_data:
            # Main angle timeline
            self._plot_angle_timeline(ax1, movement_data)
            
            # Velocity analysis
            self._plot_velocity_analysis(ax2, movement_data)
            
            # Phase distribution
            self._plot_phase_distribution(ax3, movement_data)
            
            # Statistics table
            self._create_statistics_table(ax4, session_data)
        else:
            ax1.text(0.5, 0.5, 'No movement data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_timeline_page(self, pdf: PdfPages, session_data: Dict):
        """Create movement timeline page"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5))
        fig.suptitle('Movement Timeline Analysis', fontsize=16, fontweight='bold')
        
        movement_data = session_data.get('movement_data', [])
        
        if movement_data:
            # Angle over time with phases
            self._plot_detailed_timeline(ax1, movement_data)
            
            # Quality metrics over time
            self._plot_quality_timeline(ax2, movement_data)
            
            # Movement events
            self._plot_movement_events(ax3, movement_data)
        else:
            for ax in [ax1, ax2, ax3]:
                ax.text(0.5, 0.5, 'No movement data available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_recommendations_page(self, pdf: PdfPages, session_data: Dict):
        """Create clinical recommendations page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Clinical Assessment & Recommendations', fontsize=16, fontweight='bold')
        
        # Clinical insights
        clinical_insights = session_data.get('analysis_results', {}).get('clinical_insights', {})
        final_results = session_data.get('analysis_results', {}).get('final_results', {})
        
        # Assessment summary (top-left)
        ax1.set_title('Clinical Assessment', fontweight='bold')
        ax1.axis('off')
        
        assessment_text = self._generate_assessment_text(final_results, clinical_insights)
        ax1.text(0.05, 0.95, assessment_text, transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', wrap=True)
        
        # Recommendations (top-right)
        ax2.set_title('Recommendations', fontweight='bold')
        ax2.axis('off')
        
        recommendations = clinical_insights.get('recommendations', [])
        if recommendations:
            rec_text = "Recommended interventions:\n\n"
            for i, rec in enumerate(recommendations, 1):
                rec_text += f"{i}. {rec}\n"
        else:
            rec_text = "No specific recommendations generated."
        
        ax2.text(0.05, 0.95, rec_text, transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', wrap=True)
        
        # Trend indicators (bottom-left)
        self._plot_trend_indicators(ax3, session_data)
        
        # Progress tracking template (bottom-right)
        ax4.set_title('Progress Tracking Template', fontweight='bold')
        ax4.axis('off')
        
        tracking_text = """
Track these metrics in future sessions:

□ Flexion ROM: ___°  (Target: 50°)
□ Extension ROM: ___°  (Target: 15°)
□ Movement Smoothness: ___%
□ Pain Level (1-10): ___
□ Functional Improvement: ___

Next assessment date: _______
        """
        
        ax4.text(0.05, 0.95, tracking_text, transform=ax4.transAxes, fontsize=9, 
                verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _draw_rom_gauge(self, ax, final_results: Dict):
        """Draw ROM gauge visualization"""
        ax.set_title('ROM Achievement Gauge', fontweight='bold')
        
        flexion_rom = final_results.get('flexion_rom', 0)
        extension_rom = final_results.get('extension_rom', 0)
        total_rom = final_results.get('total_rom', 0)
        
        normal_range = final_results.get('normal_range', {})
        normal_total = normal_range.get('total', 65)
        
        # Create gauge
        angles = np.linspace(0, 180, 100)
        radius = 1
        
        # Background arc
        x_bg = radius * np.cos(np.radians(angles))
        y_bg = radius * np.sin(np.radians(angles))
        ax.plot(x_bg, y_bg, 'lightgray', linewidth=8)
        
        # Achievement arc
        achievement_angle = (total_rom / normal_total) * 180
        achievement_angles = np.linspace(0, min(180, achievement_angle), 50)
        x_ach = radius * np.cos(np.radians(achievement_angles))
        y_ach = radius * np.sin(np.radians(achievement_angles))
        
        color = 'green' if total_rom >= normal_total * 0.8 else 'orange' if total_rom >= normal_total * 0.6 else 'red'
        ax.plot(x_ach, y_ach, color, linewidth=8)
        
        # Add text
        ax.text(0, -0.3, f'{total_rom:.1f}°', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0, -0.5, f'of {normal_total:.0f}° normal', ha='center', va='center', fontsize=10)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_quality_chart(self, ax, movement_quality: Dict):
        """Draw movement quality chart"""
        ax.set_title('Movement Quality Scores', fontweight='bold')
        
        qualities = ['Overall', 'Smoothness', 'Consistency']
        scores = [
            movement_quality.get('overall_score', 0),
            movement_quality.get('smoothness_score', 0),
            movement_quality.get('consistency_score', 0)
        ]
        
        colors = ['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in scores]
        
        bars = ax.barh(qualities, scores, color=colors, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score (%)')
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 2, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}%', va='center', fontweight='bold')
    
    def _draw_rom_comparison(self, ax, final_results: Dict):
        """Draw ROM comparison chart"""
        ax.set_title('ROM vs Normal Range', fontweight='bold')
        
        categories = ['Flexion', 'Extension', 'Total']
        achieved = [
            final_results.get('flexion_rom', 0),
            final_results.get('extension_rom', 0),
            final_results.get('total_rom', 0)
        ]
        
        normal_range = final_results.get('normal_range', {})
        normal = [
            normal_range.get('flexion', 50),
            normal_range.get('extension', 15),
            normal_range.get('total', 65)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, achieved, width, label='Achieved', color='skyblue', alpha=0.8)
        ax.bar(x + width/2, normal, width, label='Normal', color='lightgreen', alpha=0.8)
        
        ax.set_ylabel('Degrees')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (ach, norm) in enumerate(zip(achieved, normal)):
            ax.text(i - width/2, ach + 1, f'{ach:.1f}°', ha='center', va='bottom', fontweight='bold')
            ax.text(i + width/2, norm + 1, f'{norm:.0f}°', ha='center', va='bottom', fontweight='bold')
    
    def _plot_angle_timeline(self, ax, movement_data: List[Dict]):
        """Plot main angle timeline"""
        ax.set_title('Trunk Angle Over Time', fontweight='bold')
        
        timestamps = []
        angles = []
        
        for data_point in movement_data:
            timestamps.append(data_point.get('timestamp', 0))
            loweback_data = data_point.get('data', {}).get('loweback_analysis', {})
            angles.append(loweback_data.get('trunk_angle', 0))
        
        if timestamps:
            # Normalize timestamps to start from 0
            start_time = timestamps[0]
            timestamps = [(t - start_time) for t in timestamps]
            
            ax.plot(timestamps, angles, 'b-', linewidth=2, label='Trunk Angle')
            
            # Add reference lines
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Neutral')
            ax.axhline(y=-50, color='r', linestyle='--', alpha=0.5, label='Target Flexion')
            ax.axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Target Extension')
            
            # Fill areas
            ax.fill_between(timestamps, angles, 0, where=np.array(angles) < 0, 
                           alpha=0.3, color='red', label='Flexion')
            ax.fill_between(timestamps, angles, 0, where=np.array(angles) > 0, 
                           alpha=0.3, color='green', label='Extension')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Angle (degrees)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_velocity_analysis(self, ax, movement_data: List[Dict]):
        """Plot velocity analysis"""
        ax.set_title('Movement Velocity', fontweight='bold')
        
        angles = []
        for data_point in movement_data:
            loweback_data = data_point.get('data', {}).get('loweback_analysis', {})
            angles.append(loweback_data.get('trunk_angle', 0))
        
        if len(angles) > 1:
            velocities = np.diff(angles)
            time_points = range(len(velocities))
            
            ax.plot(time_points, velocities, 'purple', linewidth=1.5)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Angular Velocity (°/frame)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            max_vel = np.max(np.abs(velocities))
            avg_vel = np.mean(np.abs(velocities))
            ax.text(0.02, 0.98, f'Max: {max_vel:.1f}°/frame\nAvg: {avg_vel:.1f}°/frame', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_phase_distribution(self, ax, movement_data: List[Dict]):
        """Plot movement phase distribution"""
        ax.set_title('Movement Phase Distribution', fontweight='bold')
        
        phases = {}
        for data_point in movement_data:
            loweback_data = data_point.get('data', {}).get('loweback_analysis', {})
            phase = loweback_data.get('movement_phase', 'unknown')
            phases[phase] = phases.get(phase, 0) + 1
        
        if phases:
            labels = list(phases.keys())
            sizes = list(phases.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_aspect('equal')
    
    def _create_statistics_table(self, ax, session_data: Dict):
        """Create statistics table"""
        ax.set_title('Movement Statistics', fontweight='bold')
        ax.axis('off')
        
        # Extract statistics
        detailed_metrics = session_data.get('analysis_results', {}).get('detailed_metrics', {})
        final_results = session_data.get('analysis_results', {}).get('final_results', {})
        
        stats_data = [
            ['Metric', 'Value', 'Unit'],
            ['Movement Duration', f"{detailed_metrics.get('movement_duration', 0):.1f}", 'seconds'],
            ['Peak Velocity', f"{detailed_metrics.get('peak_velocity', 0):.1f}", '°/s'],
            ['Data Points', f"{detailed_metrics.get('data_points', 0)}", 'count'],
            ['Repetitions', f"{detailed_metrics.get('repetitions_completed', 0)}", 'count'],
            ['ROM Achievement', f"{final_results.get('rom_percentage', 0):.1f}", '%']
        ]
        
        table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(len(stats_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    def _plot_detailed_timeline(self, ax, movement_data: List[Dict]):
        """Plot detailed timeline with phases"""
        ax.set_title('Detailed Movement Timeline', fontweight='bold')
        
        timestamps = []
        angles = []
        phases = []
        
        for data_point in movement_data:
            timestamps.append(data_point.get('timestamp', 0))
            loweback_data = data_point.get('data', {}).get('loweback_analysis', {})
            angles.append(loweback_data.get('trunk_angle', 0))
            phases.append(loweback_data.get('movement_phase', 'unknown'))
        
        if timestamps:
            start_time = timestamps[0]
            timestamps = [(t - start_time) for t in timestamps]
            
            # Plot angle
            ax.plot(timestamps, angles, 'b-', linewidth=2, alpha=0.7)
            
            # Color-code phases
            phase_colors = {
                'neutral': 'gray',
                'flexing': 'orange', 
                'extending': 'green',
                'deep_flexion': 'red',
                'extension': 'lightgreen'
            }
            
            for i in range(len(timestamps)):
                phase = phases[i]
                color = phase_colors.get(phase, 'gray')
                ax.scatter(timestamps[i], angles[i], c=color, s=10, alpha=0.6)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Trunk Angle (degrees)')
            ax.grid(True, alpha=0.3)
            
            # Add legend for phases
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8, label=phase.title()) 
                             for phase, color in phase_colors.items() if phase in phases]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_quality_timeline(self, ax, movement_data: List[Dict]):
        """Plot quality metrics timeline"""
        ax.set_title('Quality Metrics Over Time', fontweight='bold')
        
        timestamps = []
        smoothness = []
        stability = []
        confidence = []
        
        for data_point in movement_data:
            timestamps.append(data_point.get('timestamp', 0))
            loweback_data = data_point.get('data', {}).get('loweback_analysis', {})
            quality = loweback_data.get('quality_metrics', {})
            
            smoothness.append(quality.get('movement_smoothness', 0))
            stability.append(quality.get('pose_stability', 0))
            confidence.append(quality.get('confidence_score', 0))
        
        if timestamps:
            start_time = timestamps[0]
            timestamps = [(t - start_time) for t in timestamps]
            
            ax.plot(timestamps, smoothness, 'g-', label='Smoothness', alpha=0.8)
            ax.plot(timestamps, stability, 'b-', label='Stability', alpha=0.8)
            ax.plot(timestamps, confidence, 'r-', label='Confidence', alpha=0.8)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Quality Score')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_movement_events(self, ax, movement_data: List[Dict]):
        """Plot movement events"""
        ax.set_title('Movement Events', fontweight='bold')
        
        # Extract events (phase changes, peak angles, etc.)
        timestamps = []
        events = []
        
        prev_phase = None
        for data_point in movement_data:
            timestamp = data_point.get('timestamp', 0)
            loweback_data = data_point.get('data', {}).get('loweback_analysis', {})
            phase = loweback_data.get('movement_phase', 'unknown')
            
            if phase != prev_phase and prev_phase is not None:
                timestamps.append(timestamp)
                events.append(f"Phase: {prev_phase} → {phase}")
            
            prev_phase = phase
        
        if timestamps and events:
            start_time = timestamps[0] if movement_data else 0
            timestamps = [(t - start_time) for t in timestamps]
            
            # Create event timeline
            for i, (time, event) in enumerate(zip(timestamps, events)):
                ax.barh(i, 0.5, left=time, height=0.3, alpha=0.7)
                ax.text(time + 0.25, i, event, va='center', fontsize=8)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Events')
            ax.set_ylim(-0.5, len(events) - 0.5)
        else:
            ax.text(0.5, 0.5, 'No significant events detected', ha='center', va='center', 
                   transform=ax.transAxes)
    
    def _plot_trend_indicators(self, ax, session_data: Dict):
        """Plot trend indicators"""
        ax.set_title('Trend Indicators', fontweight='bold')
        
        # This would normally compare with previous sessions
        # For now, show current session indicators
        final_results = session_data.get('analysis_results', {}).get('final_results', {})
        
        indicators = ['Flexion ROM', 'Extension ROM', 'Total ROM', 'Quality Score']
        values = [
            final_results.get('flexion_rom', 0),
            final_results.get('extension_rom', 0), 
            final_results.get('total_rom', 0),
            session_data.get('analysis_results', {}).get('movement_quality', {}).get('overall_score', 0)
        ]
        
        # Simulate trend (would come from historical data)
        trends = ['↑', '→', '↑', '↑']  # Up, stable, up, up
        colors = ['green', 'orange', 'green', 'green']
        
        bars = ax.barh(indicators, values, color=colors, alpha=0.7)
        
        # Add trend arrows
        for i, (bar, trend, value) in enumerate(zip(bars, trends, values)):
            ax.text(value + max(values) * 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{trend} {value:.1f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Value')
    
    def _generate_assessment_text(self, final_results: Dict, clinical_insights: Dict) -> str:
        """Generate clinical assessment text"""
        assessment = final_results.get('assessment', 'unknown')
        rom_percentage = final_results.get('rom_percentage', 0)
        primary_limitation = clinical_insights.get('primary_limitation')
        
        text = f"Clinical Assessment: {assessment.replace('_', ' ').title()}\n\n"
        text += f"ROM Achievement: {rom_percentage:.1f}% of normal range\n\n"
        
        if primary_limitation:
            text += f"Primary Limitation: {primary_limitation.replace('_', ' ').title()}\n\n"
        
        # Add interpretation
        if rom_percentage >= 90:
            text += "Patient demonstrates excellent range of motion with minimal limitations."
        elif rom_percentage >= 75:
            text += "Patient shows good functional range with slight limitations."
        elif rom_percentage >= 50:
            text += "Patient demonstrates moderate limitations requiring intervention."
        else:
            text += "Patient shows significant limitations requiring comprehensive treatment."
        
        return text
    
    def generate_json_report(self, session_data: Dict, output_path: str) -> bool:
        """Generate JSON format report"""
        try:
            report_data = {
                "report_generated": datetime.now().isoformat(),
                "report_type": "TriageROM Analysis",
                "version": "1.0",
                "session_summary": session_data.get('session_info', {}),
                "analysis_results": session_data.get('analysis_results', {}),
                "movement_data_summary": {
                    "total_data_points": len(session_data.get('movement_data', [])),
                    "duration_seconds": session_data.get('session_info', {}).get('duration', 0)
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error generating JSON report: {e}")
            return False