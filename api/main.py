#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import logging
from typing import Dict, List
import cv2
import numpy as np
import base64
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules with absolute imports
from utils.config_loader import ConfigLoader
from utils.logging_utils import setup_logging
from api.services.live_pose_service import LivePoseService
from api.services.loweback_analysis_service import LowerBackAnalysisService
from api.services.session_service import SessionService
from api.models.live_request_models import LiveAnalysisRequest, SessionCreateRequest
from api.models.loweback_response_models import LiveAnalysisResponse, SessionCompleteResponse
from api.websocket.connection_manager import ConnectionManager

# Initialize configuration
config_loader = ConfigLoader()
config = config_loader.merge_configs("default_config", "mediapipe_config", "loweback_config")

# Setup logging
logger = setup_logging(
    level=config.get("api", {}).get("log_level", "INFO"),
    log_file=config.get("api", {}).get("log_file")
)

# Initialize FastAPI app
app = FastAPI(
    title=config.get("api", {}).get("title", "TriageROM API"),
    version=config.get("api", {}).get("version", "1.0.0"),
    description="Range of Motion Analysis API with Real-time Pose Estimation"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pose_service = LivePoseService(config)
loweback_service = LowerBackAnalysisService(config)
session_service = SessionService(config)
connection_manager = ConnectionManager()

# Initialize pose service
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    success = pose_service.initialize()
    if not success:
        logger.error("Failed to initialize pose estimation service")
        raise RuntimeError("Pose service initialization failed")
    
    logger.info("TriageROM API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pose_service.cleanup()
    logger.info("TriageROM API shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "TriageROM API",
        "version": config.get("api", {}).get("version", "1.0.0"),
        "pose_service_ready": pose_service.is_ready()
    }

# Session management endpoints
@app.post("/api/v1/sessions/create", response_model=Dict)
async def create_session(request: SessionCreateRequest):
    """Create a new analysis session"""
    try:
        session_id = session_service.create_session(request.session_type)
        
        return {
            "status": "success",
            "session_id": session_id,
            "session_type": request.session_type,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "status": "success",
        "session": session
    }

@app.post("/api/v1/sessions/{session_id}/complete")
async def complete_session(session_id: str, background_tasks: BackgroundTasks):
    """Complete a session and generate files"""
    try:
        # Get final analysis results
        analysis_results = loweback_service.get_final_results(session_id)
        
        if not analysis_results:
            raise HTTPException(status_code=404, detail="Session not found or no analysis data")
        
        # Complete session in background
        background_tasks.add_task(
            session_service.complete_session_async,
            session_id,
            analysis_results
        )
        
        return {
            "status": "completed",
            "session_id": session_id,
            "message": "Session completed, files being generated",
            "analysis_results": analysis_results
        }
        
    except Exception as e:
        logger.error(f"Session completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Single frame analysis endpoint
@app.post("/api/v1/analyze/frame", response_model=Dict)
async def analyze_frame(request: LiveAnalysisRequest):
    """Analyze a single frame for pose and ROM"""
    try:
        # Add size validation
        # if len(request.image_data) > 10 * 1024 * 1024:  # 10MB limit
        #     raise HTTPException(status_code=413, detail="Image too large")
        # Decode base64 image
        image_data = base64.b64decode(request.image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Analyze pose
        pose_result = pose_service.analyze_frame(image)
        
        # if not pose_result.get("pose_detected", False):
        #     return {
        #         "status": "no_pose_detected",
        #         "message": "No person detected in frame",
        #         "timestamp": pose_result.get("timestamp")
        #     }
        if not pose_result.get("pose_detected", False):
            return {
                "status": "no_pose_detected",
                "message": "No person detected in frame",
                "timestamp": pose_result.get("timestamp"),
                "guidance": {
                    "instructions": [
                        "Step back to show full body",
                        "Ensure good lighting",
                        "Face the camera directly"
                    ]
                }
            }
        # Analyze lower back ROM
        rom_result = loweback_service.analyze_movement(
            pose_result,
            session_id=request.session_id
        )
        
        return rom_result
        
    except Exception as e:
        logger.error(f"Frame analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for live analysis
@app.websocket("/ws/live-analysis/{session_id}")
async def websocket_live_analysis(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time pose and ROM analysis"""
    await connection_manager.connect(websocket, session_id)
    
    try:
        # Verify session exists
        session = session_service.get_session(session_id)
        if not session:
            await websocket.send_json({
                "status": "error",
                "error_type": "SESSION_NOT_FOUND",
                "message": f"Session {session_id} not found"
            })
            await websocket.close()
            return
        
        logger.info(f"WebSocket connected for session {session_id}")
        
        while True:
            # Receive image data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "image_frame":
                try:
                    # Process the frame
                    result = await process_websocket_frame(message, session_id)
                    await websocket.send_json(result)
                    
                except Exception as e:
                    error_response = {
                        "status": "error",
                        "error_type": "PROCESSING_ERROR",
                        "message": str(e),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send_json(error_response)
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        connection_manager.disconnect(websocket, session_id)
        
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        connection_manager.disconnect(websocket, session_id)

async def process_websocket_frame(message: Dict, session_id: str) -> Dict:
    """Process a single frame from WebSocket"""
    try:
        # Decode image
        image_data = base64.b64decode(message["image_data"])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "status": "error",
                "error_type": "INVALID_IMAGE",
                "message": "Could not decode image data"
            }
        
        # Analyze pose
        pose_result = pose_service.analyze_frame(image)
        
        if not pose_result.get("pose_detected", False):
            return {
                "status": "no_pose_detected",
                "message": "No person detected in frame",
                "guidance": {
                    "instructions": [
                        "Step back to show full body",
                        "Ensure good lighting",
                        "Face the camera directly"
                    ]
                }
            }
        
        # Analyze lower back movement
        rom_result = loweback_service.analyze_movement(pose_result, session_id)
        
        # Add session activity update
        session_service.update_session_activity(session_id, rom_result)
        
        return rom_result
        
    except Exception as e:
        return {
            "status": "error",
            "error_type": "PROCESSING_ERROR",
            "message": str(e)
        }

# Statistics endpoint
@app.get("/api/v1/stats")
async def get_statistics():
    """Get API statistics"""
    return {
        "sessions": session_service.get_session_stats(),
        "pose_service": pose_service.get_service_stats(),
        "loweback_service": loweback_service.get_service_stats(),
        "websocket_connections": connection_manager.get_connection_count()
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.get("api", {}).get("host", "0.0.0.0"),
        port=config.get("api", {}).get("port", 8000),
        reload=config.get("api", {}).get("debug", False),
        log_level=config.get("api", {}).get("log_level", "info").lower()
    )