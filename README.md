How to Run the System

1. Installation
   bash# Create virtual environment
   python -m venv triagerom_env
   source triagerom_env/bin/activate # On Windows: triagerom_env\Scripts\activate

# Install dependencies

pip install -r requirements.txt 2. Directory Structure Setup
Create the directory structure and copy all the code files:
TriageROM/
├── requirements.txt
├── config/
│ ├── default_config.toml
│ ├── mediapipe_config.toml
│ └── loweback_config.toml
├── core/
├── rom_analysis/
├── data_processing/
├── api/
├── utils/
├── live_analysis/
├── examples/
└── scripts/ 3. Running Options
Option 1: Run API Server
bash# Basic run
python scripts/run_api.py

# With custom settings

python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload

# The API will be available at:

# - http://localhost:8000/docs (Swagger UI)

# - http://localhost:8000/redoc (ReDoc)

# - WebSocket: ws://localhost:8000/ws/live-analysis/{session_id}

Option 2: Run Live Analysis Example
bashpython examples/live_loweback_analysis.py
Option 3: Process Video File
bashpython scripts/process_video.py path/to/your/video.mp4 --output-dir ./results
Option 4: API Client Example
bash# Make sure API server is running first
python examples/api_client_example.py 4. API Usage Examples
Create Session and Analyze Frame (HTTP):
pythonimport requests
import base64

# Create session

response = requests.post("http://localhost:8000/api/v1/sessions/create",
json={"session_type": "lumbar_rom"})
session_id = response.json()["session_id"]

# Analyze frame

with open("image.jpg", "rb") as f:
image_data = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/api/v1/analyze/frame",
json={"image_data": image_data, "session_id": session_id})
result = response.json()
WebSocket Live Analysis:
javascriptconst ws = new WebSocket(`ws://localhost:8000/ws/live-analysis/${sessionId}`);

ws.onmessage = function(event) {
const data = JSON.parse(event.data);
console.log('Analysis result:', data);
};

// Send frame
ws.send(JSON.stringify({
type: "image_frame",
image_data: base64ImageData,
session_id: sessionId
})); 5. Configuration
Edit config/default_config.toml to customize:
toml[api]
host = "0.0.0.0"
port = 8000

[pose]
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

[loweback]
target_flexion_rom = 50.0
target_extension_rom = 15.0
enable_guidance = false

[processing]
enable_filtering = true
filter_type = "butterworth" 6. Expected Outputs
The system generates:

MOT files: Motion data compatible with OpenSim
JSON files: Complete session data
Real-time analysis: Live ROM measurements
Clinical insights: Movement quality assessment

7. Testing the System

Health Check: curl http://localhost:8000/health
Live Camera: Run the live analysis example
Video Processing: Process a test video file
API Integration: Use the client example

The system is now complete and ready for lower back ROM analysis with MediaPipe Holistic integration, comprehensive angle calculations from Sports2D, and a scalable API architecture!
