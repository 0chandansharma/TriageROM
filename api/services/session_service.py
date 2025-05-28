#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
from typing import Dict, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_processing.session_manager import SessionManager

class SessionService:
    """Service for managing analysis sessions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session_manager = SessionManager(config.get("session", {}))
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def create_session(self, session_type: str = "lumbar_rom") -> str:
        """Create a new analysis session"""
        return self.session_manager.create_session(session_type)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.session_manager.get_session(session_id)
    
    def update_session_activity(self, session_id: str, movement_data: Dict):
        """Update session with new movement data"""
        self.session_manager.update_session_activity(session_id)
        self.session_manager.add_movement_data(session_id, movement_data)
    
    def complete_session(self, session_id: str, analysis_results: Dict) -> Dict:
        """Complete a session and generate files"""
        return self.session_manager.complete_session(session_id, analysis_results)
    
    async def complete_session_async(self, session_id: str, analysis_results: Dict) -> Dict:
        """Complete session asynchronously"""
        # Run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.session_manager.complete_session,
            session_id,
            analysis_results
        )
    
    def get_session_stats(self) -> Dict:
        """Get session statistics"""
        return self.session_manager.get_session_stats()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    self.session_manager.cleanup_expired_sessions()
                    await asyncio.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    print(f"Session cleanup error: {e}")
                    await asyncio.sleep(60)  # Retry after 1 minute on error
        
        # Only start if not already running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def __del__(self):
        """Cancel cleanup task on deletion"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()