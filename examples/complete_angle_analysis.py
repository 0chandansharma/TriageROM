#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete angle analysis example showing all calculated angles
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from api.services.live_pose_service import LivePoseService
from visualization.sports2d_visualizer import Sports2DVisualizer
from visualization.angle_visualizer import AngleVisualizer

def analyze_image_with_all_angles(image_path: str, model_type: str = "mediapipe"):
    """Analyze image and show all calculated angles"""
    
    print(f"Analyzing image: {image_path}")
    print(f"Using model: {model_type}")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.merge_configs("default_config")
    
    # Set model type
    config["pose"]["model_type"] = model_type
    config["angles"]["calculate_all_angles"] = True
    config["visualization"]["draw_angles"] = True
    config["visualization"]["draw_angle_arcs"] = True
    config["visualization"]["draw_angle_labels"] = True
    
    # Initialize services
    pose_service = LivePoseService(config)
    sports2d_visualizer = Sports2DVisualizer(config)
    angle_visualizer = AngleVisualizer(config)
    
    # Initialize pose service
    if not pose_service.initialize():
        print("Failed to initialize pose service")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Analyze pose
    print("Analyzing pose...")
    pose_result = pose_service.analyze_frame(image)
    
    if not pose_result.get("pose_detected", False):
        print("No pose detected in image")
        return
    
    print("Pose detected successfully!")
    
    # Calculate all angles
    print("Calculating all angles...")
    all_keypoints = pose_result.get("all_pose_landmarks", {}) or pose_result.get("spine_keypoints", {})
    keypoint_names = list(all_keypoints.keys())
    
    # Calculate angles
    angles = angle_visualizer.calculate_all_angles(all_keypoints, keypoint_names)
    
    # Print all calculated angles
    print("\n" + "="*50)
    print("CALCULATED ANGLES")
    print("="*50)
    
    # Separate joint and segment angles
    joint_angles = {}
    segment_angles = {}
    
    for angle_name, angle_value in angles.items():
        if not np.isnan(angle_value):
            if any(word in angle_name for word in ['ankle', 'knee', 'hip', 'shoulder', 'elbow', 'wrist']):
                joint_angles[angle_name] = angle_value
            else:
                segment_angles[angle_name] = angle_value
    
    # Print joint angles
    if joint_angles:
        print("\nJOINT ANGLES:")
        print("-" * 30)
        for name, value in sorted(joint_angles.items()):
            print(f"{name:20}: {value:7.1f}°")
    
    # Print segment angles
    if segment_angles:
        print("\nSEGMENT ANGLES:")
        print("-" * 30)
        for name, value in sorted(segment_angles.items()):
            print(f"{name:20}: {value:7.1f}°")
    
    print("\n" + "="*50)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Original image
    original_with_pose = sports2d_visualizer.draw_pose_sports2d_style(
        image.copy(), pose_result, person_id=0, draw_angles=False
    )
    
    # Image with all angles
    image_with_angles = angle_visualizer.draw_all_angles_on_image(
        image.copy(), pose_result, person_id=0
    )
    
    # Combined visualization
    combined_image = sports2d_visualizer.draw_pose_sports2d_style(
        image.copy(), pose_result, person_id=0, draw_angles=False
    )
    combined_image = angle_visualizer.draw_all_angles_on_image(
        combined_image, pose_result, person_id=0
    )
    
    # Create side-by-side comparison
    h, w = image.shape[:2]
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
    
    comparison[:, :w] = original_with_pose
    comparison[:, w:w*2] = image_with_angles
    comparison[:, w*2:w*3] = combined_image
    
    # Add labels
    cv2.putText(comparison, "Original + Pose", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "All Angles", (w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Combined", (w*2 + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save results
    output_dir = Path("angle_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Save individual images
    cv2.imwrite(str(output_dir / f"{image_name}_pose_only.jpg"), original_with_pose)
    cv2.imwrite(str(output_dir / f"{image_name}_angles_only.jpg"), image_with_angles)
    cv2.imwrite(str(output_dir / f"{image_name}_combined.jpg"), combined_image)
    cv2.imwrite(str(output_dir / f"{image_name}_comparison.jpg"), comparison)
    
    # Save angle data as JSON
    angle_data = {
        "image_path": image_path,
        "model_type": model_type,
        "joint_angles": joint_angles,
        "segment_angles": segment_angles,
        "total_angles_calculated": len(joint_angles) + len(segment_angles),
        "pose_confidence": pose_result.get("service_stats", {}).get("success_rate", 0)
    }
    
    with open(output_dir / f"{image_name}_angles.json", 'w') as f:
        json.dump(angle_data, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"- Pose only: {image_name}_pose_only.jpg")
    print(f"- Angles only: {image_name}_angles_only.jpg") 
    print(f"- Combined: {image_name}_combined.jpg")
    print(f"- Comparison: {image_name}_comparison.jpg")
    print(f"- Angle data: {image_name}_angles.json")
    
    # Display results
    print("\nDisplaying results (press any key to close)...")
    
    # Resize for display if too large
    display_height = 800
    if comparison.shape[0] > display_height:
        scale = display_height / comparison.shape[0]
        new_width = int(comparison.shape[1] * scale)
        comparison = cv2.resize(comparison, (new_width, display_height))
    
    cv2.imshow("Angle Analysis Results", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Cleanup
    pose_service.cleanup()
    
    print("Analysis complete!")

def main():
    """Main function"""
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Complete angle analysis with visualization")
    # parser.add_argument("image_path", default="/Users/chandansharma/Desktop/workspace/deecogs-workspace/chandanrnd/rom-analysis-api/scripts/me1.jpg", help="Path to input image")
    # parser.add_argument("--model", default="rtmpose", 
    #                    choices=["mediapipe", "rtmpose", "openpose"],
    #                    help="Pose estimation model to use")
    
    # args = parser.parse_args()
    image_path = "/Users/chandansharma/Desktop/workspace/deecogs-workspace/chandanrnd/rom-analysis-api/scripts/me1.jpg"
    model = "rtmpose"
    # if not Path(args.image_path).exists():
    #     print(f"Image file not found: {args.image_path}")
    #     return
    
    # analyze_image_with_all_angles(args.image_path, args.model)
    analyze_image_with_all_angles(image_path, model)

if __name__ == "__main__":
    main()