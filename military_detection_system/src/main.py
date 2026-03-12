import sys
import os
import json
import threading
from urllib.parse import urlparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import MilitaryDetector
import time


def _is_stream_url(value):
    """Return True for network stream URLs (rtsp/http/https/rtmp)."""
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)

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
        
    # Update local video paths to absolute paths (leave stream URLs unchanged)
    for cam in streams:
        url = cam['url']
        if not _is_stream_url(url) and not os.path.isabs(url):
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
