#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional
from .base_estimator import BasePoseEstimator
from .mediapipe_holistic import MediaPipeHolisticEstimator
from .rtmpose_estimator import RTMPoseEstimator
from .openpose_estimator import OpenPoseEstimator

class PoseModelFactory:
    """Factory for creating pose estimation models"""
    
    @staticmethod
    def create_estimator(model_type: str, config: Dict) -> Optional[BasePoseEstimator]:
        """Create pose estimator based on model type"""
        
        model_type = model_type.lower()
        
        if model_type in ['mediapipe', 'mediapipe_holistic', 'holistic']:
            return MediaPipeHolisticEstimator(config)
        
        elif model_type in ['rtmpose', 'rtm', 'rtmlib']:
            try:
                return RTMPoseEstimator(config)
            except ImportError as e:
                print(f"RTMPose not available: {e}")
                return None
        
        elif model_type in ['openpose', 'op']:
            try:
                return OpenPoseEstimator(config)
            except ImportError as e:
                print(f"OpenPose not available: {e}")
                return None
        
        else:
            print(f"Unknown model type: {model_type}")
            return None
    
    @staticmethod
    def get_available_models() -> Dict[str, bool]:
        """Get list of available models"""
        models = {
            'mediapipe': True,  # Always available
        }
        
        # Check RTMPose
        try:
            import rtmlib
            models['rtmpose'] = True
        except ImportError:
            models['rtmpose'] = False
        
        # Check OpenPose
        try:
            import openpose
            models['openpose'] = True
        except ImportError:
            models['openpose'] = False
        
        return models