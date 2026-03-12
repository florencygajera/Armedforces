import sys
import os
import json
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import MilitaryDetector
import time

def process_camera(camera_config, detector):
    print(f"Starting inference on {camera_config['id']} ({camera_config['url']})")
    try:
        detector.process_stream(camera_config['url'], camera_config['id'])
    except Exception as e:
        print(f"Stream {camera_config['id']} stopped: {e}")

if __name__ == "__main__":
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project root: {project_root}")
    
    detector = MilitaryDetector(model_path="yolov8x.pt", confidence_thresh=0.5)

    streams_config = os.path.join(project_root, "configs", "streams_config.json")
    print(f"Loading streams from: {streams_config}")
    
    with open(streams_config, "r") as f:
        streams = json.load(f)["cameras"]
        
    # Update stream URLs to absolute paths
    for cam in streams:
        url = cam['url']
        if not os.path.isabs(url):
            cam['url'] = os.path.join(project_root, url)
        print(f"Camera {cam['id']}: {cam['url']}")

    threads = []
    for cam in streams:
        t = threading.Thread(target=process_camera, args=(cam, detector), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system.")
