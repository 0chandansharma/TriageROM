#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os

class VideoUtils:
    """Video processing utilities"""
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video information"""
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_width: int, target_height: int, 
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """Resize frame with optional aspect ratio maintenance"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        
        if maintain_aspect_ratio:
            # Calculate scaling factor
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Create canvas and center the resized frame
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_offset = (target_height - new_h) // 2
            x_offset = (target_width - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(frame, (target_width, target_height))
    
    @staticmethod
    def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int],
                                font_scale: float = 0.7, thickness: int = 2,
                                text_color: Tuple[int, int, int] = (255, 255, 255),
                                bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Draw text with background rectangle"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        x, y = position
        cv2.rectangle(frame, 
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + baseline + 5),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
        
        return frame
    
    @staticmethod
    def create_video_writer(output_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
        """Create video writer with appropriate codec"""
        # Try different codecs
        codecs = ['mp4v', 'XVID', 'MJPG']
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if writer.isOpened():
                return writer
            else:
                writer.release()
        
        # If all fail, try default
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))