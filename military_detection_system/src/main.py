import json
import threading
from src.detector import MilitaryDetector
import time

def process_camera(camera_config, detector):
    print(f"Starting inference on {camera_config['id']} ({camera_config['url']})")
    try:
        detector.process_stream(camera_config['url'], camera_config['id'])
    except Exception as e:
        print(f"Stream {camera_config['id']} stopped: {e}")

if __name__ == "__main__":
    detector = MilitaryDetector(model_path="yolov8x.pt", confidence_thresh=0.5)

    with open("configs/streams_config.json", "r") as f:
        streams = json.load(f)["cameras"]

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
