#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import WebSocket
from typing import Dict, List, Set
import asyncio
import json
import time

class ConnectionManager:
    """Manage WebSocket connections for live analysis"""
    
    def __init__(self):
        # Active connections by session
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        
        # Statistics
        self.total_connections = 0
        self.current_connections = 0
        self.messages_sent = 0
        self.messages_failed = 0
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        # Add to active connections
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "session_id": session_id,
            "connected_at": time.time(),
            "last_activity": time.time(),
            "messages_sent": 0,
            "messages_received": 0
        }
        
        # Update statistics
        self.total_connections += 1
        self.current_connections += 1
        
        print(f"WebSocket connected for session {session_id}. Total active: {self.current_connections}")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection"""
        # Remove from active connections
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            
            # Clean up empty session list
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        
        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        # Update statistics
        self.current_connections -= 1
        
        print(f"WebSocket disconnected for session {session_id}. Total active: {self.current_connections}")
    
    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
            
            # Update metadata
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_sent"] += 1
                self.connection_metadata[websocket]["last_activity"] = time.time()
            
            self.messages_sent += 1
            
        except Exception as e:
            print(f"Failed to send message to WebSocket: {e}")
            self.messages_failed += 1
            
            # Connection might be dead, try to clean it up
            await self._cleanup_dead_connection(websocket)
    
    async def send_to_session(self, message: Dict, session_id: str):
        """Send message to all connections in a session"""
        if session_id not in self.active_connections:
            return
        
        # Send to all connections for this session
        dead_connections = []
        
        for websocket in self.active_connections[session_id]:
            try:
                await websocket.send_text(json.dumps(message))
                
                # Update metadata
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    self.connection_metadata[websocket]["last_activity"] = time.time()
                
                self.messages_sent += 1
                
            except Exception as e:
                print(f"Failed to send message to WebSocket in session {session_id}: {e}")
                self.messages_failed += 1
                dead_connections.append(websocket)
        
        # Clean up dead connections
        for websocket in dead_connections:
            await self._cleanup_dead_connection(websocket)
    
    async def broadcast_message(self, message: Dict):
        """Broadcast message to all active connections"""
        all_connections = []
        for session_connections in self.active_connections.values():
            all_connections.extend(session_connections)
        
        dead_connections = []
        
        for websocket in all_connections:
            try:
                await websocket.send_text(json.dumps(message))
                
                # Update metadata
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    self.connection_metadata[websocket]["last_activity"] = time.time()
                
                self.messages_sent += 1
                
            except Exception as e:
                print(f"Failed to broadcast message to WebSocket: {e}")
                self.messages_failed += 1
                dead_connections.append(websocket)
        
        # Clean up dead connections
        for websocket in dead_connections:
            await self._cleanup_dead_connection(websocket)
    
    async def _cleanup_dead_connection(self, websocket: WebSocket):
        """Clean up a dead WebSocket connection"""
        try:
            # Find which session this connection belongs to
            session_id = None
            if websocket in self.connection_metadata:
                session_id = self.connection_metadata[websocket]["session_id"]
            else:
                # Search through all sessions
                for sid, connections in self.active_connections.items():
                    if websocket in connections:
                        session_id = sid
                        break
            
            if session_id:
                self.disconnect(websocket, session_id)
            
            # Try to close the connection
            await websocket.close()
            
        except Exception as e:
            print(f"Error cleaning up dead connection: {e}")
    
    def get_session_connections(self, session_id: str) -> List[WebSocket]:
        """Get all connections for a session"""
        return self.active_connections.get(session_id, [])
    
    def get_connection_count(self) -> Dict:
        """Get connection count statistics"""
        session_counts = {
            session_id: len(connections) 
            for session_id, connections in self.active_connections.items()
        }
        
        return {
            "total_current_connections": self.current_connections,
            "total_lifetime_connections": self.total_connections,
            "active_sessions": len(self.active_connections),
            "connections_by_session": session_counts,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed
        }
    
    def get_connection_info(self, websocket: WebSocket) -> Dict:
        """Get information about a specific connection"""
        if websocket not in self.connection_metadata:
            return {"error": "Connection not found"}
        
        metadata = self.connection_metadata[websocket].copy()
        metadata["connection_age_seconds"] = time.time() - metadata["connected_at"]
        metadata["last_activity_seconds_ago"] = time.time() - metadata["last_activity"]
        
        return metadata
    
    async def ping_all_connections(self):
        """Send ping to all connections to check if they're alive"""
        ping_message = {
            "type": "ping",
            "timestamp": time.time()
        }
        
        await self.broadcast_message(ping_message)
    
    async def cleanup_stale_connections(self, max_idle_seconds: int = 300):
        """Clean up connections that haven't been active recently"""
        current_time = time.time()
        stale_connections = []
        
        for websocket, metadata in self.connection_metadata.items():
            if current_time - metadata["last_activity"] > max_idle_seconds:
                stale_connections.append(websocket)
        
        for websocket in stale_connections:
            await self._cleanup_dead_connection(websocket)
        
        return len(stale_connections)