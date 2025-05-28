#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import threading
import time

class CameraManager:
    """Manage camera input and video streaming"""
    
    def __init__(self, camera_id: int = 0, config: Dict = None):
        self.camera_id = camera_id
        self.config = config or {}
        
        # Camera properties
        self.cap = None
        self.is_opened = False
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30.0
        
        # Threading for continuous capture
        self.capture_thread = None
        self.stop_capture = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_fps_update = time.time()
        self.actual_fps = 0.0
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties
            target_width = self.config.get('camera_width', 640)
            target_height = self.config.get('camera_height', 480)
            target_fps = self.config.get('camera_fps', 30)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            
            # Get actual properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def start_capture(self) -> bool:
        """Start continuous frame capture in separate thread"""
        if not self.is_opened:
            return False
        
        if self.capture_thread and self.capture_thread.is_alive():
            return True  # Already running
        
        self.stop_capture = False
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def stop_capture(self):
        """Stop continuous frame capture"""
        self.stop_capture = True
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
    
    def _capture_loop(self):
        """Continuous capture loop running in separate thread"""
        frame_count = 0
        last_time = time.time()
        
        while not self.stop_capture and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                self.frames_captured += 1
                frame_count += 1
                
                # Update FPS calculation
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    self.actual_fps = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time
                
            else:
                self.frames_dropped += 1
                time.sleep(0.01)  # Brief pause if frame capture fails
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame (synchronous)"""
        if not self.is_opened or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.frames_captured += 1
            return frame
        else:
            self.frames_dropped += 1
            return None
    
    def get_camera_info(self) -> Dict:
        """Get camera information"""
        return {
            "camera_id": self.camera_id,
            "is_opened": self.is_opened,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "target_fps": self.fps,
            "actual_fps": round(self.actual_fps, 2),
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "capture_active": self.capture_thread and self.capture_thread.is_alive()
        }
    
    def adjust_camera_settings(self, **settings):
        """Adjust camera settings dynamically"""
        if not self.cap:
            return False
        
        success = True
        
        if 'brightness' in settings:
            success &= self.cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'])
        
        if 'contrast' in settings:
            success &= self.cap.set(cv2.CAP_PROP_CONTRAST, settings['contrast'])
        
        if 'exposure' in settings:
            success &= self.cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
        
        if 'gain' in settings:
            success &= self.cap.set(cv2.CAP_PROP_GAIN, settings['gain'])
        
        return success
    
    def cleanup(self):
        """Cleanup camera resources"""
        self.stop_capture()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_opened = False
        print("Camera resources cleaned up")