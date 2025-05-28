#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

class SpineKeypoint(BaseModel):
    """Individual spine keypoint model"""
    name: str
    x: float = Field(..., ge=0, le=1, description="Normalized x coordinate")
    y: float = Field(..., ge=0, le=1, description="Normalized y coordinate") 
    z: float = Field(default=0.0, description="Z coordinate")
    visibility: float = Field(..., ge=0, le=1, description="Keypoint visibility score")
    world_x: float = Field(default=0.0, description="World X coordinate")
    world_y: float = Field(default=0.0, description="World Y coordinate")
    world_z: float = Field(default=0.0, description="World Z coordinate")

class SpineKeypoints(BaseModel):
    """Spine keypoints collection"""
    required_points: List[str]
    landmarks: List[SpineKeypoint]

class RangeTracking(BaseModel):
    """Range of motion tracking data"""
    max_flexion: float = Field(..., description="Maximum flexion angle (negative)")
    max_extension: float = Field(..., description="Maximum extension angle (positive)")
    current_rom: float = Field(..., description="Current total range of motion")
    target_rom: float = Field(..., description="Target/expected range of motion")

class QualityMetrics(BaseModel):
    """Movement quality metrics"""
    movement_smoothness: float = Field(..., ge=0, le=1, description="Movement smoothness score")
    compensatory_movement: float = Field(..., ge=0, le=1, description="Compensatory movement level")
    pose_stability: float = Field(..., ge=0, le=1, description="Pose detection stability")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence score")

class LowerBackAnalysis(BaseModel):
    """Lower back analysis results"""
    trunk_angle: float = Field(..., description="Current trunk angle in degrees")
    hip_angle: Optional[float] = Field(default=None, description="Hip angle in degrees")
    net_spine_angle: Optional[float] = Field(default=None, description="Net spinal movement angle")
    movement_phase: str = Field(..., description="Current movement phase")
    direction: str = Field(..., description="Movement direction")
    range_tracking: RangeTracking
    quality_metrics: QualityMetrics

class RealTimeFeedback(BaseModel):
    """Real-time user feedback"""
    instruction: str = Field(..., description="Current instruction for user")
    progress_percentage: float = Field(..., ge=0, le=100, description="Movement progress percentage")
    form_cues: List[str] = Field(default=[], description="Form correction cues")
    warning: Optional[str] = Field(default=None, description="Warning message if any")

class LiveAnalysisResponse(BaseModel):
    """Live analysis response model"""
    status: str = Field(..., description="Analysis status")
    timestamp: float = Field(..., description="Analysis timestamp")
    session_id: str = Field(..., description="Session identifier")
    movement_type: str = Field(default="lumbar_flexion_extension", description="Type of movement being analyzed")
    current_state: str = Field(..., description="Current analysis state")
    
    spine_keypoints: SpineKeypoints
    loweback_analysis: LowerBackAnalysis
    real_time_feedback: Optional[RealTimeFeedback] = Field(default=None, description="User guidance feedback")

class FinalResults(BaseModel):
    """Final ROM analysis results"""
    flexion_rom: float = Field(..., description="Achieved flexion range of motion")
    extension_rom: float = Field(..., description="Achieved extension range of motion")
    total_rom: float = Field(..., description="Total range of motion")
    normal_range: Dict[str, float] = Field(..., description="Normal range values for comparison")
    rom_percentage: float = Field(..., description="ROM as percentage of normal")
    assessment: str = Field(..., description="Clinical assessment category")

class MovementQuality(BaseModel):
    """Movement quality assessment"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    smoothness_score: float = Field(..., ge=0, le=100, description="Movement smoothness score")
    consistency_score: float = Field(..., ge=0, le=100, description="Movement consistency score")

class DetailedMetrics(BaseModel):
    """Detailed movement metrics"""
    movement_duration: float = Field(..., description="Total movement duration in seconds")
    peak_velocity: float = Field(..., description="Peak movement velocity")
    data_points: int = Field(..., description="Number of data points collected")
    repetitions_completed: int = Field(..., description="Number of complete repetitions")

class MOTFileInfo(BaseModel):
    """MOT file generation information"""
    file_generated: bool = Field(..., description="Whether MOT file was generated")
    file_path: Optional[str] = Field(default=None, description="Path to generated MOT file")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    data_points: Optional[int] = Field(default=None, description="Number of data points in file")
    sampling_rate: Optional[float] = Field(default=None, description="Data sampling rate")

class ClinicalInsights(BaseModel):
    """Clinical insights and recommendations"""
    primary_limitation: Optional[str] = Field(default=None, description="Primary movement limitation")
    compensation_pattern: Optional[str] = Field(default=None, description="Observed compensation pattern")
    recommendations: List[str] = Field(default=[], description="Clinical recommendations")

class SessionCompleteResponse(BaseModel):
    """Session completion response model"""
    status: str = Field(default="completed", description="Completion status")
    session_id: str = Field(..., description="Session identifier")
    movement_type: str = Field(..., description="Type of movement analyzed")
    completion_timestamp: float = Field(..., description="Completion timestamp")
    
    final_results: FinalResults
    movement_quality: MovementQuality
    detailed_metrics: DetailedMetrics
    mot_file_info: MOTFileInfo
    clinical_insights: ClinicalInsights

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = Field(default="error", description="Response status")
    error_type: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Error message")
    timestamp: float = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    guidance: Optional[Dict[str, List[str]]] = Field(default=None, description="User guidance for fixing error")