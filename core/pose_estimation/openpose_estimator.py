#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_estimator import BasePoseEstimator

try:
    import openpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False

class OpenPoseEstimator(BasePoseEstimator):
    """OpenPose estimation implementation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not OPENPOSE_AVAILABLE:
            raise ImportError("OpenPose not available. Please install OpenPose Python API")
        
        self.opWrapper = None
        self.model_folder = config.get('model_folder', './models/')
        self.face = config.get('face', False)
        self.hand = config.get('hand', False)
        
        # OpenPose COCO 18-point skeleton connections
        self.skeleton_connections = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),  # Upper body
            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),  # Lower body
            (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)  # Head
        ]
        
    def initialize(self) -> bool:
        """Initialize OpenPose"""
        try:
            # Set parameters
            params = dict()
            params["model_folder"] = self.model_folder
            params["face"] = self.face
            params["hand"] = self.hand
            
            # Initialize OpenPose
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize OpenPose: {e}")
            return False
    
    def estimate_pose(self, image: np.ndarray) -> Dict:
        """Estimate pose using OpenPose"""
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        try:
            # Create datum
            datum = op.Datum()
            datum.cvInputData = image
            
            # Process
            self.opWrapper.emplaceAndPop([datum])
            
            # Get results
            if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
                return {
                    "pose_detected": False,
                    "processing_successful": True,
                    "spine_keypoints": {},
                    "all_pose_landmarks": {}
                }
            
            # Format results (take first person)
            person_keypoints = datum.poseKeypoints[0]  # Shape: (18, 3) - x, y, confidence
            
            formatted_results = self._format_openpose_results(person_keypoints, image.shape)
            
            return formatted_results
            
        except Exception as e:
            return {"error": f"OpenPose estimation failed: {str(e)}"}
    
    def _format_openpose_results(self, keypoints: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict:
        """Format OpenPose results to standard format"""
        height, width = image_shape[:2]
        
        # OpenPose COCO keypoint names
        keypoint_names = [
            "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
            "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "left_eye", "right_ear", "left_ear"
        ]
        
        formatted_data = {
            "pose_detected": True,
            "processing_successful": True,
            "spine_keypoints": {},
            "all_pose_landmarks": {},
            "model_type": "openpose_coco18"
        }
        
        # Process all keypoints
        for i, (name) in enumerate(keypoint_names):
            if i >= len(keypoints):
                break
                
            x, y, confidence = keypoints[i]
            
            if confidence > 0:  # Valid keypoint
                formatted_data["all_pose_landmarks"][name] = {
                    "x": float(x) / width,
                    "y": float(y) / height,
                    "z": 0.0,
                    "visibility": float(confidence),
                    "pixel_x": int(x),
                    "pixel_y": int(y)
                }
        
        # Extract spine-specific keypoints with OpenPose naming
        spine_mapping = {
            "nose": "nose",
            "left_shoulder": "left_shoulder", 
            "right_shoulder": "right_shoulder",
            "left_hip": "left_hip",
            "right_hip": "right_hip"
        }
        
        for std_name, op_name in spine_mapping.items():
            if op_name in formatted_data["all_pose_landmarks"]:
                formatted_data["spine_keypoints"][std_name] = formatted_data["all_pose_landmarks"][op_name]
        
        return formatted_data
    
    def cleanup(self):
        """Cleanup OpenPose resources"""
        if self.opWrapper:
            self.opWrapper.stop()
        self.is_initialized = False