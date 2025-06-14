#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced demo script to test TriageROM API with live video display and output saving
"""

import requests
import base64
import json
import cv2
import time
import numpy as np
from pathlib import Path
import threading
from queue import Queue

class TriageROMAPIDemo:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session_id = None
        self.show_live = True
        self.save_output_video = True
        
    def test_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ API is healthy")
                return True
            else:
                print(f"✗ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to API: {e}")
            return False
    
    def create_session(self):
        """Create a new analysis session"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/sessions/create",
                json={"session_type": "lumbar_rom"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                print(f"✓ Session created: {self.session_id}")
                return True
            else:
                print(f"✗ Session creation failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Session creation error: {e}")
            return False
    
    def draw_analysis_overlay(self, frame, analysis_result):
        """Draw analysis overlay on frame"""
        if not analysis_result:
            return frame
        
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Colors (BGR format)
        colors = {
            'good': (0, 255, 0),        # Green
            'warning': (0, 255, 255),   # Yellow
            'error': (0, 0, 255),       # Red
            'text': (255, 255, 255),    # White
            'bg': (0, 0, 0)             # Black
        }
        
        status = analysis_result.get("status", "unknown")
        
        if status == "analyzing":
            # Draw pose keypoints if available
            spine_keypoints = analysis_result.get("spine_keypoints", {})
            if "landmarks" in spine_keypoints:
                self._draw_pose_landmarks(overlay_frame, spine_keypoints["landmarks"], width, height)
            
            # Draw analysis info panel
            loweback_data = analysis_result.get("loweback_analysis", {})
            if loweback_data:
                self._draw_analysis_panel(overlay_frame, loweback_data)
                
                # Draw trunk angle visualization
                trunk_angle = loweback_data.get("trunk_angle", 0)
                self._draw_trunk_angle_indicator(overlay_frame, trunk_angle)
        
        elif status == "no_pose_detected":
            # Draw guidance message
            cv2.putText(overlay_frame, "No Pose Detected", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors['warning'], 3)
            cv2.putText(overlay_frame, "Stand back and show full body", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['warning'], 2)
            
            guidance = analysis_result.get("guidance", {}).get("instructions", [])
            for i, instruction in enumerate(guidance):
                cv2.putText(overlay_frame, f"• {instruction}", (50, 200 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 1)
        
        else:
            # Draw error status
            message = analysis_result.get("message", "Processing...")
            cv2.putText(overlay_frame, f"Status: {status}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors['error'], 2)
            cv2.putText(overlay_frame, message, (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['error'], 2)
        
        # Add frame info
        cv2.putText(overlay_frame, f"Session: {self.session_id[-8:] if self.session_id else 'None'}", 
                   (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        cv2.putText(overlay_frame, f"Time: {time.strftime('%H:%M:%S')}", 
                   (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        cv2.putText(overlay_frame, "Press 'q' to quit, 's' to save screenshot", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)
        
        return overlay_frame
    
    def _draw_pose_landmarks(self, frame, landmarks, width, height):
        """Draw pose landmarks on frame"""
        colors = {
            'keypoint': (0, 255, 0),     # Green
            'connection': (0, 255, 255), # Yellow
            'low_conf': (128, 128, 128)  # Gray
        }
        
        # Convert landmarks to pixel coordinates
        keypoints_dict = {}
        for landmark in landmarks:
            name = landmark.get('name')
            if name:
                x = int(landmark.get('x', 0) * width)
                y = int(landmark.get('y', 0) * height)
                visibility = landmark.get('visibility', 0)
                keypoints_dict[name] = {'x': x, 'y': y, 'visibility': visibility}
        
        # Draw connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip')
        ]
        
        for conn in connections:
            if conn[0] in keypoints_dict and conn[1] in keypoints_dict:
                pt1 = keypoints_dict[conn[0]]
                pt2 = keypoints_dict[conn[1]]
                if pt1['visibility'] > 0.5 and pt2['visibility'] > 0.5:
                    cv2.line(frame, (pt1['x'], pt1['y']), (pt2['x'], pt2['y']), 
                            colors['connection'], 2)
        
        # Draw keypoints
        for name, kp in keypoints_dict.items():
            if kp['visibility'] > 0.5:
                color = colors['keypoint']
                radius = 4
            else:
                color = colors['low_conf']
                radius = 2
            
            cv2.circle(frame, (kp['x'], kp['y']), radius, color, -1)
            cv2.circle(frame, (kp['x'], kp['y']), radius + 1, (0, 0, 0), 1)
    
    def _draw_analysis_panel(self, frame, loweback_data):
        """Draw analysis information panel"""
        # Panel background
        panel_x, panel_y = 10, 10
        panel_width, panel_height = 300, 200
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ROM Analysis", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = 50
        
        # Current measurements
        trunk_angle = loweback_data.get('trunk_angle', 0)
        movement_phase = loweback_data.get('movement_phase', 'unknown')
        direction = loweback_data.get('direction', 'unknown')
        
        measurements = [
            f"Trunk Angle: {trunk_angle:.1f}°",
            f"Phase: {movement_phase.replace('_', ' ').title()}",
            f"Direction: {direction.replace('_', ' ').title()}"
        ]
        
        for measurement in measurements:
            cv2.putText(frame, measurement, (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # ROM tracking
        range_tracking = loweback_data.get('range_tracking', {})
        if range_tracking:
            y_offset += 10
            max_flexion = range_tracking.get('max_flexion', 0)
            max_extension = range_tracking.get('max_extension', 0)
            current_rom = range_tracking.get('current_rom', 0)
            target_rom = range_tracking.get('target_rom', 65)
            
            rom_info = [
                f"Max Flexion: {max_flexion:.1f}°",
                f"Max Extension: {max_extension:.1f}°",
                f"Current ROM: {current_rom:.1f}°"
            ]
            
            for info in rom_info:
                cv2.putText(frame, info, (panel_x + 10, panel_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
            
            # Progress bar
            progress = min(100, (current_rom / target_rom) * 100) if target_rom > 0 else 0
            self._draw_progress_bar(frame, (panel_x + 10, panel_y + y_offset), 
                                  200, 15, progress)
    
    def _draw_trunk_angle_indicator(self, frame, trunk_angle):
        """Draw trunk angle indicator gauge"""
        center_x = frame.shape[1] - 100
        center_y = 100
        radius = 50
        
        # Background circle
        cv2.circle(frame, (center_x, center_y), radius, (60, 60, 60), 2)
        
        # Angle indicator
        angle_rad = np.radians(trunk_angle + 90)  # Offset for vertical reference
        end_x = int(center_x + radius * 0.8 * np.cos(angle_rad))
        end_y = int(center_y + radius * 0.8 * np.sin(angle_rad))
        
        # Color based on angle
        if -10 <= trunk_angle <= 10:
            color = (0, 255, 0)  # Green - neutral
        elif -45 <= trunk_angle <= 25:
            color = (0, 255, 255)  # Yellow - acceptable
        else:
            color = (0, 0, 255)  # Red - extreme
        
        cv2.line(frame, (center_x, center_y), (end_x, end_y), color, 3)
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
        
        # Angle text
        cv2.putText(frame, f"{trunk_angle:.1f}°", (center_x - 30, center_y + radius + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_progress_bar(self, frame, position, width, height, progress):
        """Draw progress bar"""
        x, y = position
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
        
        # Progress fill
        fill_width = int((progress / 100.0) * width)
        if fill_width > 0:
            color = (0, 255, 0) if progress >= 80 else (0, 255, 255) if progress >= 50 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
        
        # Progress text
        cv2.putText(frame, f"{progress:.1f}%", (x + width + 5, y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def analyze_video_file_with_display(self, video_path, save_output=True):
        """Analyze video file with live display and optional output saving"""
        if not self.session_id:
            print("✗ No active session")
            return False
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Cannot open video: {video_path}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing {total_frames} frames at {fps} FPS ({frame_width}x{frame_height})...")
        
        # Setup output video writer if saving
        output_writer = None
        if save_output:
            output_path = Path(video_path).parent / f"{Path(video_path).stem}_triagerom_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
            print(f"Saving output video to: {output_path}")
        
        # Initialize display window
        window_name = "TriageROM Live Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        results = []
        frame_count = 0
        successful_analyses = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Encode frame to base64 (process every frame for smooth display)
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    image_data = base64.b64encode(buffer).decode('utf-8')
                    
                    # Analyze frame
                    result = self.analyze_frame(image_data)
                    
                    # Draw overlay
                    display_frame = self.draw_analysis_overlay(frame, result)
                    
                    # Save successful analyses
                    if result and result.get("status") == "analyzing":
                        successful_analyses += 1
                        results.append({
                            "frame": frame_count,
                            "timestamp": frame_count / fps,
                            "analysis": result
                        })
                        
                        # Print progress for successful analyses
                        if successful_analyses % 30 == 0:
                            loweback = result.get("loweback_analysis", {})
                            trunk_angle = loweback.get("trunk_angle", 0)
                            rom_data = loweback.get("range_tracking", {})
                            current_rom = rom_data.get("current_rom", 0)
                            print(f"Frame {frame_count}: Angle={trunk_angle:.1f}°, ROM={current_rom:.1f}°")
                    
                    # Add frame counter to display
                    progress = (frame_count / total_frames) * 100
                    cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Save frame to output video
                    if output_writer:
                        output_writer.write(display_frame)
                    
                    # Display frame
                    cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = Path(video_path).parent / f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(str(screenshot_path), display_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord(' '):
                    # Pause/resume
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('r'):
                    # Reset session (create new one)
                    print("Resetting session...")
                    if self.create_session():
                        results = []
                        successful_analyses = 0
        
        finally:
            cap.release()
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
        
        print(f"✓ Processed {frame_count} frames, {successful_analyses} successful analyses")
        
        # Get final session results
        final_results = self.complete_session()
        
        # Save results
        output_data = {
            "video_info": {
                "path": str(video_path),
                "total_frames": total_frames,
                "fps": fps,
                "processed_frames": frame_count,
                "successful_analyses": successful_analyses,
                "output_video": str(output_path) if save_output else None
            },
            "final_results": final_results,
            "frame_results": results[-100:]  # Last 100 results
        }
        
        results_file = Path(video_path).parent / f"{Path(video_path).stem}_triagerom_results.json"
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Results saved to: {results_file}")
        if save_output:
            print(f"✓ Output video saved to: {output_path}")
        
        # Print summary
        if final_results and "final_results" in final_results:
            fr = final_results["final_results"]
            print(f"\n=== ROM Analysis Summary ===")
            print(f"Flexion ROM: {fr.get('flexion_rom', 0):.1f}°")
            print(f"Extension ROM: {fr.get('extension_rom', 0):.1f}°")
            print(f"Total ROM: {fr.get('total_rom', 0):.1f}°")
            print(f"Assessment: {fr.get('assessment', 'Unknown')}")
            print(f"ROM Achievement: {fr.get('rom_percentage', 0):.1f}%")
        
        return True
    
    def analyze_frame(self, image_data):
        """Analyze a single frame"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/analyze/frame",
                json={
                    "image_data": image_data,
                    "session_id": self.session_id,
                    "timestamp": time.time()
                },
                timeout=5  # Reduced timeout for real-time processing
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Network error: {str(e)}"}
    
    def complete_session(self):
        """Complete the analysis session"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/sessions/{self.session_id}/complete",
                timeout=30
            )
            
            if response.status_code == 200:
                print("✓ Session completed successfully")
                return response.json()
            else:
                print(f"Session completion failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Session completion error: {e}")
            return None
    
    def test_webcam_live(self, camera_id=0):
        """Test with live webcam feed"""
        if not self.create_session():
            return False
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"✗ Cannot open camera {camera_id}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        window_name = "TriageROM Live Webcam"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        print("Live webcam analysis started...")
        print("Controls: 'q' to quit, 's' to save screenshot, SPACE to pause")
        
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame_count += 1
                    
                    # Process every 3rd frame for performance
                    if frame_count % 3 == 0:
                        # Encode and analyze frame
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        image_data = base64.b64encode(buffer).decode('utf-8')
                        result = self.analyze_frame(image_data)
                    else:
                        result = {"status": "processing", "message": "Skipped frame for performance"}
                    
                    # Draw overlay
                    display_frame = self.draw_analysis_overlay(frame, result)
                    
                    # Add webcam info
                    cv2.putText(display_frame, "LIVE WEBCAM", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"webcam_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.complete_session()
        
        print("✓ Webcam analysis completed")
        return True

def main():
    """Run the enhanced demo"""
    print("TriageROM API Enhanced Demo")
    print("===========================")
    print("Features:")
    print("- Live video display with pose overlay")
    print("- Save output video with analysis")
    print("- Real-time ROM visualization")
    print("- Interactive controls")
    print()
    
    # Initialize demo
    demo = TriageROMAPIDemo()
    
    # Test API connection
    if not demo.test_health():
        print("Please start the TriageROM API server first:")
        print("python run_api_server.py")
        return
    
    # Menu
    print("Select demo mode:")
    print("1. Analyze video file with live display")
    print("2. Live webcam analysis")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Video file analysis
        video_file = input("Enter video file path (or press Enter for default): ").strip()
        if not video_file:
            video_file = "/Users/chandansharma/Desktop/workspace/deecogs-workspace/TriageROM/demo/demo-flexion-1.mov"
        
        video_path = Path(video_file)
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            # Try in current directory
            video_path = Path.cwd() / video_file
            if not video_path.exists():
                print(f"Video file not found: {video_path}")
                return
        
        print(f"\nAnalyzing video: {video_path}")
        print("Controls during playback:")
        print("- 'q': Quit")
        print("- 's': Save screenshot") 
        print("- SPACE: Pause/Resume")
        print("- 'r': Reset session")
        print()
        
        if demo.create_session():
            demo.analyze_video_file_with_display(video_path, save_output=True)
    
    elif choice == "2":
        # Webcam analysis
        camera_id = input("Enter camera ID (default 0): ").strip()
        camera_id = int(camera_id) if camera_id.isdigit() else 0
        
        print(f"\nStarting webcam analysis with camera {camera_id}")
        demo.test_webcam_live(camera_id)
    
    elif choice == "3":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()