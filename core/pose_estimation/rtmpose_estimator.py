#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_estimator import BasePoseEstimator

try:
    from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False

class RTMPoseEstimator(BasePoseEstimator):
    """RTMPose estimation implementation using RTMLib"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not RTMLIB_AVAILABLE:
            raise ImportError("RTMLib not available. Install with: pip install rtmlib")
        
        self.pose_tracker = None
        self.model_type = config.get('model_type', 'body_with_feet')
        self.mode = config.get('mode', 'balanced')
        self.det_frequency = config.get('det_frequency', 1)
        
        # Set up skeleton connections based on model type
        self._setup_skeleton_connections()
        
    def _setup_skeleton_connections(self):
        """Setup skeleton connections based on model type"""
        if self.model_type == 'body_with_feet':
            # HALPE_26 connections
            self.skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (12, 14), (13, 15), (14, 16),  # Legs
                (15, 19), (15, 20), (16, 22), (16, 23),  # Feet
                (19, 20), (22, 23)  # Toes
            ]
        elif self.model_type == 'body':
            # COCO_17 connections
            self.skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
            ]
        else:
            # Default connections
            self.skeleton_connections = []
    
    def initialize(self) -> bool:
        """Initialize RTMPose model"""
        try:
            # Select model class
            if self.model_type == 'body_with_feet':
                ModelClass = BodyWithFeet
            elif self.model_type == 'whole_body':
                ModelClass = Wholebody
            elif self.model_type == 'body':
                ModelClass = Body
            elif self.model_type == 'hand':
                ModelClass = Hand
            else:
                ModelClass = BodyWithFeet
            
            # Initialize pose tracker
            self.pose_tracker = PoseTracker(
                ModelClass,
                det_frequency=self.det_frequency,
                mode=self.mode,
                backend='onnxruntime',
                device='cpu',
                tracking=False,
                to_openpose=False
            )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize RTMPose: {e}")
            return False
    
    def estimate_pose(self, image: np.ndarray) -> Dict:
        """Estimate pose using RTMPose"""
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        try:
            # Get pose estimation results
            keypoints, scores = self.pose_tracker(image)
            
            if len(keypoints) == 0:
                return {
                    "pose_detected": False,
                    "processing_successful": True,
                    "spine_keypoints": {},
                    "all_pose_landmarks": {}
                }
            
            # Format results (take first person)
            person_keypoints = keypoints[0]  # Shape: (num_keypoints, 2)
            person_scores = scores[0]  # Shape: (num_keypoints,)
            
            formatted_results = self._format_rtmpose_results(
                person_keypoints, person_scores, image.shape
            )
            
            return formatted_results
            
        except Exception as e:
            return {"error": f"RTMPose estimation failed: {str(e)}"}
    
    def _format_rtmpose_results(self, keypoints: np.ndarray, scores: np.ndarray, 
                               image_shape: Tuple[int, int, int]) -> Dict:
        """Format RTMPose results to standard format"""
        height, width = image_shape[:2]
        
        # Get keypoint names based on model type
        if self.model_type == 'body_with_feet':
            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle",
                "head", "neck", "left_big_toe", "left_small_toe", "left_heel",
                "right_big_toe", "right_small_toe", "right_heel", "background"
            ]
        elif self.model_type == 'body':
            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
        else:
            keypoint_names = [f"point_{i}" for i in range(len(keypoints))]
        
        formatted_data = {
            "pose_detected": True,
            "processing_successful": True,
            "spine_keypoints": {},
            "all_pose_landmarks": {},
            "model_type": self.model_type
        }
        
        # Process all keypoints
        for i, (kpt, score, name) in enumerate(zip(keypoints, scores, keypoint_names)):
            if i >= len(keypoint_names):
                break
                
            formatted_data["all_pose_landmarks"][name] = {
                "x": float(kpt[0]) / width,
                "y": float(kpt[1]) / height,
                "z": 0.0,
                "visibility": float(score),
                "pixel_x": int(kpt[0]),
                "pixel_y": int(kpt[1])
            }
        
        # Extract spine-specific keypoints
        spine_keypoint_names = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        for name in spine_keypoint_names:
            if name in formatted_data["all_pose_landmarks"]:
                formatted_data["spine_keypoints"][name] = formatted_data["all_pose_landmarks"][name]
        
        return formatted_data
    
    def cleanup(self):
        """Cleanup RTMPose resources"""
        self.pose_tracker = None
        self.is_initialized = False