import requests
import base64
import json

# Create session
response = requests.post("http://localhost:8000/api/v1/sessions/create", 
                        json={"session_type": "lumbar_rom"})
session_id = response.json()["session_id"]

# Analyze frame with all angles enabled
with open("/Users/chandansharma/Desktop/workspace/deecogs-workspace/chandanrnd/rom-analysis-api/scripts/me1.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/api/v1/analyze/frame", 
                        json={
                            "image_data": image_data, 
                            "session_id": session_id,
                            "calculate_all_angles": True
                        })

result = response.json()

# All angles will be in the response
if "all_calculated_angles" in result:
    angles = result["all_calculated_angles"]
    print("Joint Angles:", angles.get("joint_angles", {}))
    print("Segment Angles:", angles.get("segment_angles", {}))