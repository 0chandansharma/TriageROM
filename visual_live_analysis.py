#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Live analysis with visual display showing pose and angles
"""

import cv2
import numpy as np
import time
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import ConfigLoader
from core.pose_estimation.mediapipe_holistic import MediaPipeHolisticEstimator
from rom_analysis.spine.lumbar.lumbar_rom_analyzer import LumbarROMAnalyzer
from live_analysis.camera_manager import CameraManager

def draw_pose_landmarks(image, pose_data):
    """Draw pose landmarks on image"""
    if not pose_data.get("pose_detected", False):
        return image
    
    spine_keypoints = pose_data.get("spine_keypoints", {})
    
    # Draw keypoints
    for kp_name, kp_data in spine_keypoints.items():
        if isinstance(kp_data, dict) and kp_data.get('visibility', 0) > 0.5:
            x = int(kp_data['pixel_x']) if 'pixel_x' in kp_data else int(kp_data['x'] * image.shape[1])
            y = int(kp_data['pixel_y']) if 'pixel_y' in kp_data else int(kp_data['y'] * image.shape[0])
            
            # Color based on visibility
            visibility = kp_data.get('visibility', 0)
            color = (0, int(255 * visibility), int(255 * (1 - visibility)))
            
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.putText(image, kp_name[:4], (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw skeleton connections
    connections = [
        ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip')
    ]
    
    for start_kp, end_kp in connections:
        if start_kp in spine_keypoints and end_kp in spine_keypoints:
            start_data = spine_keypoints[start_kp]
            end_data = spine_keypoints[end_kp]
            
            if (isinstance(start_data, dict) and isinstance(end_data, dict) and
                start_data.get('visibility', 0) > 0.5 and end_data.get('visibility', 0) > 0.5):
                
                start_x = int(start_data['pixel_x']) if 'pixel_x' in start_data else int(start_data['x'] * image.shape[1])
                start_y = int(start_data['pixel_y']) if 'pixel_y' in start_data else int(start_data['y'] * image.shape[0])
                end_x = int(end_data['pixel_x']) if 'pixel_x' in end_data else int(end_data['x'] * image.shape[1])
                end_y = int(end_data['pixel_y']) if 'pixel_y' in end_data else int(end_data['y'] * image.shape[0])
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    return image

def draw_rom_info(image, rom_analysis):
    """Draw ROM analysis information on image"""
    if not rom_analysis or rom_analysis.get("status") != "analyzing":
        return image
    
    loweback_data = rom_analysis.get("loweback_analysis", {})
    
    # Background for text
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 255)  # Yellow
    thickness = 2
    
    # Current measurements
    trunk_angle = loweback_data.get("trunk_angle", 0)
    cv2.putText(image, f"Trunk Angle: {trunk_angle:.1f}°", (20, y_offset), font, font_scale, color, thickness)
    
    y_offset += 25
    movement_phase = loweback_data.get("movement_phase", "unknown")
    cv2.putText(image, f"Phase: {movement_phase}", (20, y_offset), font, font_scale, color, thickness)
    
    y_offset += 25
    direction = loweback_data.get("direction", "unknown")
    cv2.putText(image, f"Direction: {direction}", (20, y_offset), font, font_scale, color, thickness)
    
    # ROM tracking
    range_tracking = loweback_data.get("range_tracking", {})
    y_offset += 30
    max_flexion = range_tracking.get("max_flexion", 0)
    cv2.putText(image, f"Max Flexion: {max_flexion:.1f}°", (20, y_offset), font, font_scale, (255, 0, 0), thickness)
    
    y_offset += 25
    max_extension = range_tracking.get("max_extension", 0)
    cv2.putText(image, f"Max Extension: {max_extension:.1f}°", (20, y_offset), font, font_scale, (0, 255, 0), thickness)
    
    y_offset += 25
    current_rom = range_tracking.get("current_rom", 0)
    target_rom = range_tracking.get("target_rom", 65)
    rom_percent = (current_rom / target_rom) * 100 if target_rom > 0 else 0
    cv2.putText(image, f"ROM: {current_rom:.1f}° ({rom_percent:.1f}%)", (20, y_offset), font, font_scale, (255, 255, 0), thickness)
    
    # Quality metrics
    quality_metrics = loweback_data.get("quality_metrics", {})
    y_offset += 30
    confidence = quality_metrics.get("confidence_score", 0)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (20, y_offset), font, font_scale, (255, 255, 255), thickness)
    
    # Draw ROM progress bar
    bar_x, bar_y = 20, image.shape[0] - 60
    bar_width, bar_height = 300, 20
    
    # Background bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Progress bar
    progress_width = int((rom_percent / 100) * bar_width)
    if progress_width > 0:
        bar_color = (0, 255, 0) if rom_percent >= 80 else (0, 255, 255) if rom_percent >= 60 else (0, 165, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), bar_color, -1)
    
    # Bar border
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.putText(image, "ROM Progress", (bar_x, bar_y - 5), font, 0.5, (255, 255, 255), 1)
    
    return image

def main():
    """Main function for visual live analysis"""
    print("TriageROM Visual Live Analysis")
    print("==============================")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.merge_configs("default_config", "mediapipe_config", "loweback_config")
    
    # Initialize components
    pose_estimator = MediaPipeHolisticEstimator(config.get("pose", {}))
    if not pose_estimator.initialize():
        print("Failed to initialize pose estimator")
        return
    
    rom_analyzer = LumbarROMAnalyzer(config.get("loweback", {}))
    camera_manager = CameraManager(
        camera_id=config.get('camera', {}).get('camera_id', 0),
        config=config.get('camera', {})
    )
    
    # Initialize camera
    if not camera_manager.initialize_camera():
        print("Failed to initialize camera")
        return
    
    print("Camera and pose estimation ready!")
    print("\nInstructions:")
    print("- Stand in front of the camera showing your full body")
    print("- Perform slow lower back flexion and extension movements")
    print("- Press 'q' to quit, 'r' to reset analysis, 's' to save screenshot")
    
    # Start session
    session_id = f"visual_session_{int(time.time())}"
    rom_analyzer.start_session()
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            frame = camera_manager.capture_single_frame()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Estimate pose
            pose_result = pose_estimator.estimate_pose(frame)
            
            # Draw pose landmarks
            frame = draw_pose_landmarks(frame, pose_result)
            
            # Analyze ROM if pose detected
            rom_result = None
            if pose_result.get("pose_detected", False):
                rom_result = rom_analyzer.analyze_movement(pose_result)
                
                # Draw ROM information
                frame = draw_rom_info(frame, rom_result)
            else:
                # Show "No pose detected" message
                cv2.putText(frame, "No pose detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Stand back and show full body", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add frame counter and FPS
            cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('TriageROM Visual Analysis', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset analysis
                rom_analyzer.reset_analysis()
                print("Analysis reset")
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
            
            # Check for movement completion
            if rom_result and rom_analyzer.detect_movement_completion():
                print("\nMovement completed! Final results:")
                final_results = rom_analyzer.get_rom_summary()
                print(json.dumps(final_results, indent=2))
                
                # Reset for next movement
                rom_analyzer.reset_analysis()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        camera_manager.cleanup()
        pose_estimator.cleanup()
        cv2.destroyAllWindows()
        print("Visual analysis complete!")

if __name__ == "__main__":
    main()