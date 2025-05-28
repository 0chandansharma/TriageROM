#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process video files for ROM analysis
"""

import sys
import cv2
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from core.pose_estimation.mediapipe_holistic import MediaPipeHolisticEstimator
from rom_analysis.spine.lumbar.lumbar_rom_analyzer import LumbarROMAnalyzer
from data_processing.session_manager import SessionManager

def process_video(video_path: str, output_dir: str, config: dict):
    """Process a video file for ROM analysis"""
    
    print(f"Processing video: {video_path}")
    
    # Initialize components
    pose_estimator = MediaPipeHolisticEstimator(config.get("pose", {}))
    if not pose_estimator.initialize():
        print("Failed to initialize pose estimator")
        return False
    
    rom_analyzer = LumbarROMAnalyzer(config.get("loweback", {}))
    session_manager = SessionManager({"data_directory": output_dir})
    
    # Create session  
    session_id = session_manager.create_session("video_analysis")
    rom_analyzer.start_session()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames at {fps} FPS")
    
    # Process frames
    frame_count = 0
    results = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Estimate pose
            pose_result = pose_estimator.estimate_pose(frame)
            
            # Analyze ROM if pose detected
            if pose_result.get("pose_detected", False):
                rom_result = rom_analyzer.analyze_movement(pose_result)
                results.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "rom_analysis": rom_result
                })
                
                # Store in session
                session_manager.add_movement_data(session_id, rom_result)
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        cap.release()
    
    # Get final results
    final_results = rom_analyzer.get_rom_summary()
    
    # Complete session and generate files
    completion_result = session_manager.complete_session(session_id, final_results)
    
    # Save detailed results
    output_file = Path(output_dir) / f"{session_id}_detailed_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "video_path": video_path,
            "total_frames": frame_count,
            "fps": fps,
            "duration": frame_count / fps,
            "final_results": final_results,
            "frame_results": results,
            "generated_files": completion_result
        }, f, indent=2, default=str)
    
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Final results saved to: {output_file}")
    
    if completion_result.get("files_generated"):
        print("Generated files:")
        for file_path in completion_result["files_generated"]:
            print(f"  - {file_path}")
    
    # Print summary
    if final_results and not final_results.get("error"):
        print(f"\nROM Analysis Summary:")
        final_res = final_results.get("final_results", {})
        print(f"  Flexion ROM: {final_res.get('flexion_rom', 0):.1f}°")
        print(f"  Extension ROM: {final_res.get('extension_rom', 0):.1f}°")
        print(f"  Total ROM: {final_res.get('total_rom', 0):.1f}°")
        print(f"  Assessment: {final_res.get('assessment', 'N/A')}")
    
    # Cleanup
    pose_estimator.cleanup()
    
    return True

def main():
    """Main function for video processing script"""
    parser = argparse.ArgumentParser(description="Process video for ROM analysis")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--output-dir", default="./video_analysis_output", 
                       help="Output directory for results")
    parser.add_argument("--config", default="default_config", 
                       help="Configuration file name")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video_path).exists():
        print(f"Video file not found: {args.video_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.merge_configs(args.config, "mediapipe_config", "loweback_config")
    
    # Process video
    success = process_video(args.video_path, str(output_dir), config)
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed!")

if __name__ == "__main__":
    main()