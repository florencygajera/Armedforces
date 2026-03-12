import cv2
import json

def test_streams(config_path="configs/streams_config.json"):
    with open(config_path, "r") as f:
        streams = json.load(f)["cameras"]
        
    for cam in streams:
        cap = cv2.VideoCapture(cam["url"])
        if not cap.isOpened():
            print(f"FAILED to open stream {cam['id']} at {cam['url']}")
        else:
            ret, frame = cap.read()
            if ret:
                print(f"SUCCESS receiving from {cam['id']}. Frame size: {frame.shape}")
            else:
                print(f"FAILED to grab frame from {cam['id']}")
        cap.release()

if __name__ == "__main__":
    test_streams()
