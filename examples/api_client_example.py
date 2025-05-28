#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example API client for TriageROM API
"""

import requests
import json
import base64
import cv2
import time
import websocket
import threading
from pathlib import Path

class TriageROMClient:
    """Client for TriageROM API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.ws = None
        
    def create_session(self, session_type: str = "lumbar_rom") -> bool:
        """Create a new analysis session"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/sessions/create",
                json={"session_type": session_type}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                print(f"Created session: {self.session_id}")
                return True
            else:
                print(f"Failed to create session: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def analyze_frame(self, image_path: str) -> dict:
        """Analyze a single frame"""
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = requests.post(
                f"{self.base_url}/api/v1/analyze/frame",
                json={
                    "image_data": image_data,
                    "session_id": self.session_id,
                    "timestamp": time.time()
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.text}"}
                
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}
    
    def complete_session(self) -> dict:
        """Complete the current session"""
        if not self.session_id:
            return {"error": "No active session"}
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/sessions/{self.session_id}/complete"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.text}"}
                
        except Exception as e:
            return {"error": f"Completion error: {str(e)}"}
    
    def start_websocket_analysis(self, camera_id: int = 0):
        """Start WebSocket-based live analysis"""
        if not self.session_id:
            print("No active session. Create a session first.")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Could not open camera {camera_id}")
            return
        
        # WebSocket URL
        ws_url = f"ws://localhost:8000/ws/live-analysis/{self.session_id}"
        
        def on_message(ws, message):
            """Handle WebSocket messages"""
            try:
                data = json.loads(message)
                
                if data.get("status") == "analyzing":
                    # Print analysis results
                    loweback = data.get("loweback_analysis", {})
                    trunk_angle = loweback.get("trunk_angle", 0)
                    phase = loweback.get("movement_phase", "unknown")
                    rom = loweback.get("range_tracking", {}).get("current_rom", 0)
                    
                    print(f"\rAngle: {trunk_angle:6.1f}° | Phase: {phase:12} | ROM: {rom:5.1f}°", end="")
                    
                elif data.get("status") == "error":
                    print(f"\nError: {data.get('message')}")
                
            except Exception as e:
                print(f"\nWebSocket message error: {e}")
        
        def on_error(ws, error):
            print(f"\nWebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"\nWebSocket closed: {close_status_code} - {close_msg}")
        
        def on_open(ws):
            print("WebSocket connected, starting live analysis...")
            
            def send_frames():
                """Send camera frames via WebSocket"""
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        # Encode frame
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_data = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send frame
                        message = {
                            "type": "image_frame",
                            "image_data": image_data,
                            "session_id": self.session_id,
                            "timestamp": time.time()
                        }
                        
                        ws.send(json.dumps(message))
                        
                        # Display frame
                        cv2.imshow('Live Analysis', frame)
                        
                        # Check for quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        time.sleep(0.033)  # ~30 FPS
                
                except Exception as e:
                    print(f"Frame sending error: {e}")
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    ws.close()
            
            # Start frame sending in separate thread
            frame_thread = threading.Thread(target=send_frames)
            frame_thread.daemon = True
            frame_thread.start()
        
        # Create and start WebSocket
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        print("Press 'q' in camera window to quit")
        self.ws.run_forever()

def main():
    """Example usage of TriageROM API client"""
    print("TriageROM API Client Example")
    print("============================")
    
    # Create client
    client = TriageROMClient("http://localhost:8000")
    
    print("1. Testing single frame analysis...")
    
    # Create session
    if not client.create_session("lumbar_rom"):
        print("Failed to create session")
        return
    
    # Test single frame analysis (you need to provide an image)
    image_path = "test_image.jpg"  # Replace with actual image path
    if Path(image_path).exists():
        result = client.analyze_frame(image_path)
        print("Analysis result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image {image_path} not found, skipping single frame test")
    
    print("\n2. Testing live WebSocket analysis...")
    print("Make sure your camera is connected and API server is running")
    
    # Start live analysis
    try:
        client.start_websocket_analysis(camera_id=0)
    except KeyboardInterrupt:
        print("\nLive analysis interrupted")
    
    # Complete session
    print("\n3. Completing session...")
    completion_result = client.complete_session()
    print("Completion result:")
    print(json.dumps(completion_result, indent=2))

if __name__ == "__main__":
    main()