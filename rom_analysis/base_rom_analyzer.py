#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

class BaseROMAnalyzer(ABC):
    """Base class for Range of Motion analyzers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.movement_history = []
        self.session_active = False
        self.session_start_time = None
        
    @abstractmethod
    def analyze_movement(self, pose_data: Dict) -> Dict:
        """
        Analyze movement from pose data
        
        Args:
            pose_data: Processed pose estimation results
            
        Returns:
            Movement analysis results
        """
        pass
    
    @abstractmethod
    def get_rom_summary(self) -> Dict:
        """
        Get summary of range of motion analysis
        
        Returns:
            ROM analysis summary
        """
        pass
    
    @abstractmethod
    def detect_movement_completion(self) -> bool:
        """
        Detect if a complete movement cycle has been performed
        
        Returns:
            True if movement is complete
        """
        pass
    
    def start_session(self):
        """Start a new analysis session"""
        self.session_active = True
        self.movement_history = []
        import time
        self.session_start_time = time.time()
    
    def end_session(self):
        """End current analysis session"""
        self.session_active = False
    
    def reset_analysis(self):
        """Reset analysis state"""
        self.movement_history = []
        self.session_active = False
        self.session_start_time = None
    
    def get_session_info(self) -> Dict:
        """Get current session information"""
        import time
        current_time = time.time()
        
        return {
            "session_active": self.session_active,
            "session_duration": current_time - self.session_start_time if self.session_start_time else 0,
            "data_points": len(self.movement_history),
            "analysis_type": self.__class__.__name__
        }