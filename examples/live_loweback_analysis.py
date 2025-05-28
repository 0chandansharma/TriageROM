#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for live lower back ROM analysis using camera
"""

import cv2
import time
import json
from pathlib import Path
import sys

# Add parent directory to path to import TriageROM modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from live_analysis.live_processor import LiveProcessor

def display_frame_callback(frame):
    """Callback to display processed frame"""
    # Add timestamp overlay
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Live Lower Back Analysis', frame)

def result_callback(result):
    """Callback to handle analysis results"""
    if result.get("status") == "analyzing":
        # Print key metrics
        loweback_data = result.get("loweback_analysis", {})
        trunk_angle = loweback_data.get("trunk_angle")
        movement_phase = loweback_data.get("movement_phase")
        rom_data = loweback_data.get("range_tracking", {})
        
        print(f"\rTrunk Angle: {trunk_angle:6.1f}° | Phase: {movement_phase:12} | ROM: {rom_data.get('current_rom', 0):5.1f}°", end="")
    
    elif result.get("status") == "error":
        print(f"\nError: {result.get('message', 'Unknown error')}")

def main():
    """Main function for live analysis example"""
    print("TriageROM Live Lower Back Analysis")
    print("==================================")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.merge_configs("default_config", "mediapipe_config", "loweback_config")
    
    # Override some settings for live analysis
    config["pose"]["min_detection_confidence"] = 0.5
    config["pose"]["min_tracking_confidence"] = 0.5
    config["loweback"]["enable_guidance"] = True
    
    # Initialize live processor
    processor = LiveProcessor(config)
    
    print("Initializing camera and pose estimation...")
    if not processor.initialize():
        print("Failed to initialize live processor")
        return
    
    print("Camera and pose estimation ready!")
    print("\nInstructions:")
    print("- Stand in front of the camera showing your full body")
    print("- Perform slow lower back flexion and extension movements")
    print("- Press 'q' to quit, 's' to start new session, 'r' to reset current session")
    print("- Press SPACE to pause/resume")
    
    session_id = f"live_session_{int(time.time())}"
    paused = False
    
    # Start processing
    if not processor.start_processing(
        session_id=session_id,
        frame_callback=display_frame_callback,
        result_callback=result_callback
    ):
        print("Failed to start live processing")
        return
    
    print(f"\nStarted live analysis session: {session_id}")
    print("Processing frames... (press 'q' to quit)")
    
    try:
        while True:
            if not paused:
                # Process frame
                result = processor.process_frame()
                
                # Check for movement completion
                if result and result.get("movement_completed", False):
                    print("\n\nMovement completed! Final results:")
                    final_summary = result.get("final_summary", {})
                    if final_summary:
                        print(json.dumps(final_summary, indent=2))
                    
                    # Ask if user wants to continue
                    print("\nPress 'c' to continue with new movement or 'q' to quit")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Start new session
                processor.stop_processing()
                session_id = f"live_session_{int(time.time())}"
                processor.start_processing(
                    session_id=session_id,
                    frame_callback=display_frame_callback,
                    result_callback=result_callback
                )
                print(f"\nStarted new session: {session_id}")
            elif key == ord('r'):
                # Reset current session
                processor.loweback_service.reset_session(session_id)
                print("\nSession reset")
            elif key == ord(' '):
                # Pause/resume
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
            elif key == ord('c'):
                # Continue (when movement completed)
                continue
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Stop processing and cleanup
        print("\nStopping analysis...")
        final_results = processor.stop_processing()
        
        if final_results.get("final_results"):
            print("\nFinal Analysis Results:")
            print(json.dumps(final_results["final_results"], indent=2))
        
        # Show processing statistics
        stats = final_results.get("processing_stats", {})
        if stats:
            print(f"\nProcessing Statistics:")
            print(f"- Total frames processed: {stats.get('total_frames_processed', 0)}")
            print(f"- Success rate: {stats.get('success_rate_percent', 0):.1f}%")
            print(f"- Average processing time: {stats.get('average_processing_time_ms', 0):.1f}ms")
            print(f"- Camera FPS: {stats.get('camera_info', {}).get('actual_fps', 0):.1f}")
        
        processor.cleanup()
        cv2.destroyAllWindows()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()