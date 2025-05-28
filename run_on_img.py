# Example of how to use different models

from utils.config_loader import ConfigLoader
from api.services.live_pose_service import LivePoseService
from visualization.sports2d_visualizer import Sports2DVisualizer

# Load config
config_loader = ConfigLoader()
config = config_loader.load_config("default_config")

# Try different models
models_to_test = ["mediapipe", "rtmpose"]

for model_type in models_to_test:
    print(f"\nTesting {model_type}...")
    
    # Update config for this model
    config["pose"]["model_type"] = model_type
    
    # Initialize service
    pose_service = LivePoseService(config)
    
    if pose_service.initialize():
        print(f"{model_type} initialized successfully!")
        
        # Test with an image
        import cv2
        image = cv2.imread("/Users/chandansharma/Desktop/workspace/deecogs-workspace/chandanrnd/rom-analysis-api/scripts/me1.jpg")
        
        if image is not None:
            result = pose_service.analyze_frame(image)
            
            if result.get("pose_detected"):
                print(f"{model_type} detected pose successfully!")
                
                # Visualize with Sports2D style
                visualizer = Sports2DVisualizer()
                vis_image = visualizer.draw_pose_sports2d_style(
                    image.copy(), result, person_id=0, draw_angles=True
                )
                
                cv2.imshow(f"{model_type} Result", vis_image)
                cv2.waitKey(10000)
            else:
                print(f"{model_type} failed to detect pose")
        
        pose_service.cleanup()
    else:
        print(f"Failed to initialize {model_type}")

cv2.destroyAllWindows()