#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_estimator import BasePoseEstimator

class MediaPipeHolisticEstimator(BasePoseEstimator):
    """MediaPipe Holistic pose estimation implementation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = None
        
    def initialize(self) -> bool:
        """Initialize MediaPipe Holistic model"""
        try:
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=self.config.get('static_image_mode', False),
                model_complexity=self.config.get('model_complexity', 1),
                smooth_landmarks=self.config.get('smooth_landmarks', True),
                enable_segmentation=self.config.get('enable_segmentation', False),
                smooth_segmentation=True,
                refine_face_landmarks=self.config.get('refine_face_landmarks', False),
                min_detection_confidence=self.config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
            )
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize MediaPipe Holistic: {e}")
            return False
    
    def estimate_pose(self, image: np.ndarray) -> Dict:
        """
        Estimate pose using MediaPipe Holistic
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            Dictionary containing pose landmarks and metadata
        """
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Process the image
        results = self.holistic.process(rgb_image)
        
        # Convert back to BGR for drawing
        rgb_image.flags.writeable = True
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Extract and format landmarks
        formatted_results = self._format_results(results, image.shape)
        
        return formatted_results
    
    def _format_results(self, results, image_shape: Tuple[int, int, int]) -> Dict:
        """Format MediaPipe results into standardized format"""
        height, width = image_shape[:2]
        
        formatted_data = {
            "pose_detected": False,
            "spine_keypoints": {},
            "all_pose_landmarks": {},
            "face_landmarks": {},
            "left_hand_landmarks": {},
            "right_hand_landmarks": {},
            "world_landmarks": {},
            "processing_successful": True
        }
        
        # Process pose landmarks
        if results.pose_landmarks:
            formatted_data["pose_detected"] = True
            
            # Extract all pose landmarks
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = self._get_landmark_name(idx)
                formatted_data["all_pose_landmarks"][landmark_name] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                    "pixel_x": int(landmark.x * width),
                    "pixel_y": int(landmark.y * height)
                }
            
            # Extract spine-specific keypoints
            spine_keypoint_indices = {
                "nose": 0,
                "left_shoulder": 11,
                "right_shoulder": 12,
                "left_hip": 23,
                "right_hip": 24
            }
            
            for kp_name, idx in spine_keypoint_indices.items():
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    formatted_data["spine_keypoints"][kp_name] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                        "pixel_x": int(landmark.x * width),
                        "pixel_y": int(landmark.y * height),
                        "world_x": 0.0,  # Will be filled from world landmarks
                        "world_y": 0.0,
                        "world_z": 0.0
                    }
        
        # Process world landmarks for 3D coordinates
        if results.pose_world_landmarks:
            spine_keypoint_indices = {
                "nose": 0,
                "left_shoulder": 11,
                "right_shoulder": 12,
                "left_hip": 23,
                "right_hip": 24
            }
            
            for kp_name, idx in spine_keypoint_indices.items():
                if idx < len(results.pose_world_landmarks.landmark) and kp_name in formatted_data["spine_keypoints"]:
                    world_landmark = results.pose_world_landmarks.landmark[idx]
                    formatted_data["spine_keypoints"][kp_name].update({
                        "world_x": world_landmark.x,
                        "world_y": world_landmark.y,
                        "world_z": world_landmark.z
                    })
        
        # Process hand landmarks
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                formatted_data["left_hand_landmarks"][f"point_{idx}"] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "pixel_x": int(landmark.x * width),
                    "pixel_y": int(landmark.y * height)
                }
        
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                formatted_data["right_hand_landmarks"][f"point_{idx}"] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "pixel_x": int(landmark.x * width),
                    "pixel_y": int(landmark.y * height)
                }
        
        return formatted_data
    
    def _get_landmark_name(self, idx: int) -> str:
        """Convert MediaPipe landmark index to name"""
        landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]
        
        if idx < len(landmark_names):
            return landmark_names[idx]
        return f"landmark_{idx}"
    
    def cleanup(self):
        """Cleanup MediaPipe resources"""
        if self.holistic:
            self.holistic.close()
        self.is_initialized = False