#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

class PoseTracker:
    """
    Pose tracking and person identification
    Adapted from Sports2D tracking logic
    """
    
    def __init__(self, max_history: int = 30, max_distance_threshold: float = 0.1):
        self.max_history = max_history
        self.max_distance_threshold = max_distance_threshold
        self.pose_history = deque(maxlen=max_history)
        self.person_id = 0
        self.tracking_active = False
        
    def track_pose(self, current_pose: Dict) -> Dict:
        """
        Track pose across frames and maintain consistency
        
        Args:
            current_pose: Current pose detection results
            
        Returns:
            Tracked pose with consistency information
        """
        if not current_pose.get("pose_detected", False):
            # No pose detected
            tracking_result = {
                "pose_detected": False,
                "tracking_active": False,
                "person_id": None,
                "tracking_confidence": 0.0,
                "message": "No pose detected"
            }
            self.tracking_active = False
            return tracking_result
        
        # First detection or reinitializing tracking
        if not self.tracking_active or len(self.pose_history) == 0:
            self.person_id += 1
            self.tracking_active = True
            tracking_result = self._create_tracking_result(current_pose, is_new_person=True)
        else:
            # Compare with previous poses
            similarity_score = self._calculate_pose_similarity(current_pose)
            
            if similarity_score > 0.7:  # Good match
                tracking_result = self._create_tracking_result(current_pose, is_new_person=False)
            elif similarity_score > 0.4:  # Moderate match
                tracking_result = self._create_tracking_result(current_pose, is_new_person=False)
                tracking_result["tracking_confidence"] = similarity_score
                tracking_result["message"] = "Moderate tracking confidence"
            else:
                # Poor match - might be new person or tracking lost
                self.person_id += 1
                tracking_result = self._create_tracking_result(current_pose, is_new_person=True)
                tracking_result["message"] = "Tracking lost, new person detected"
        
        # Store current pose in history
        self.pose_history.append(self._extract_pose_signature(current_pose))
        
        return tracking_result
    
    def _create_tracking_result(self, pose: Dict, is_new_person: bool) -> Dict:
        """Create tracking result with metadata"""
        result = pose.copy()
        result.update({
            "tracking_active": True,
            "person_id": self.person_id,
            "is_new_person": is_new_person,
            "tracking_confidence": 0.9 if not is_new_person else 1.0,
            "message": "New person detected" if is_new_person else "Tracking active"
        })
        return result
    
    def _calculate_pose_similarity(self, current_pose: Dict) -> float:
        """
        Calculate similarity between current pose and recent history
        Based on keypoint positions and proportions
        """
        if not self.pose_history:
            return 0.0
        
        current_signature = self._extract_pose_signature(current_pose)
        if not current_signature:
            return 0.0
        
        # Compare with recent poses (last 5 frames)
        recent_signatures = list(self.pose_history)[-5:]
        similarities = []
        
        for past_signature in recent_signatures:
            similarity = self._compare_pose_signatures(current_signature, past_signature)
            if similarity is not None:
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _extract_pose_signature(self, pose: Dict) -> Optional[Dict]:
        """Extract pose signature for comparison"""
        spine_keypoints = pose.get("spine_keypoints", {})
        
        if len(spine_keypoints) < 3:
            return None
        
        # Extract key measurements
        signature = {}
        
        # Get keypoint positions
        keypoint_positions = {}
        for kp_name, kp_data in spine_keypoints.items():
            if kp_data.get('visibility', 0) > 0.5:
                keypoint_positions[kp_name] = (kp_data['x'], kp_data['y'])
        
        if len(keypoint_positions) < 3:
            return None
        
        # Calculate relative distances and ratios
        if 'left_shoulder' in keypoint_positions and 'right_shoulder' in keypoint_positions:
            shoulder_width = abs(keypoint_positions['left_shoulder'][0] - 
                               keypoint_positions['right_shoulder'][0])
            signature['shoulder_width'] = shoulder_width
        
        if 'left_hip' in keypoint_positions and 'right_hip' in keypoint_positions:
            hip_width = abs(keypoint_positions['left_hip'][0] - 
                          keypoint_positions['right_hip'][0])
            signature['hip_width'] = hip_width
        
        # Torso length
        if ('left_shoulder' in keypoint_positions and 'right_shoulder' in keypoint_positions and
            'left_hip' in keypoint_positions and 'right_hip' in keypoint_positions):
            
            shoulder_mid_y = (keypoint_positions['left_shoulder'][1] + 
                            keypoint_positions['right_shoulder'][1]) / 2
            hip_mid_y = (keypoint_positions['left_hip'][1] + 
                        keypoint_positions['right_hip'][1]) / 2
            
            signature['torso_length'] = abs(hip_mid_y - shoulder_mid_y)
        
        # Center position
        if keypoint_positions:
            center_x = np.mean([pos[0] for pos in keypoint_positions.values()])
            center_y = np.mean([pos[1] for pos in keypoint_positions.values()])
            signature['center'] = (center_x, center_y)
        
        return signature
    
    def _compare_pose_signatures(self, sig1: Dict, sig2: Dict) -> Optional[float]:
        """Compare two pose signatures and return similarity score"""
        if not sig1 or not sig2:
            return None
        
        similarities = []
        
        # Compare proportional measurements
        for measurement in ['shoulder_width', 'hip_width', 'torso_length']:
            if measurement in sig1 and measurement in sig2:
                val1, val2 = sig1[measurement], sig2[measurement]
                if val1 > 0 and val2 > 0:
                    ratio = min(val1, val2) / max(val1, val2)
                    similarities.append(ratio)
        
        # Compare center positions (should be relatively stable)
        if 'center' in sig1 and 'center' in sig2:
            center1, center2 = sig1['center'], sig2['center']
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            # Convert distance to similarity (closer = more similar)
            position_similarity = max(0, 1 - distance / self.max_distance_threshold)
            similarities.append(position_similarity)
        
        return np.mean(similarities) if similarities else None
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.pose_history.clear()
        self.tracking_active = False
        self.person_id = 0
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            "tracking_active": self.tracking_active,
            "current_person_id": self.person_id,
            "history_length": len(self.pose_history),
            "max_history": self.max_history
        }