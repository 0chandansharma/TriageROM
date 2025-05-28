#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class LiveAnalysisRequest(BaseModel):
    """Request model for live frame analysis"""
    image_data: str = Field(..., description="Base64 encoded image data")
    session_id: str = Field(..., description="Session identifier")
    timestamp: Optional[float] = Field(default=None, description="Client timestamp")
    image_format: Optional[str] = Field(default="jpg", description="Image format (jpg, png)")
    
    class Config:
        schema_extra = {
            "example": {
                "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ...",
                "session_id": "lumbar_rom_1234567890_abcd1234",
                "timestamp": 1642781234.567,
                "image_format": "jpg"
            }
        }

class SessionCreateRequest(BaseModel):
    """Request model for creating analysis session"""
    session_type: str = Field(default="lumbar_rom", description="Type of analysis session")
    patient_id: Optional[str] = Field(default=None, description="Patient identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional session metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "session_type": "lumbar_rom",
                "patient_id": "patient_001",
                "metadata": {
                    "therapist": "Dr. Smith",
                    "notes": "Follow-up assessment"
                }
            }
        }

class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str = Field(..., description="Message type")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Message data")
    timestamp: Optional[float] = Field(default=None, description="Message timestamp")

class ImageFrameMessage(WebSocketMessage):
    """Image frame message for WebSocket"""
    type: str = Field(default="image_frame", description="Message type")
    image_data: str = Field(..., description="Base64 encoded image data")
    session_id: str = Field(..., description="Session identifier")