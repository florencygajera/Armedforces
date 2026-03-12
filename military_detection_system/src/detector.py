from ultralytics import YOLO
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alerts import AlertSystem
from src.stream_handler import RTSPStreamHandler


class MilitaryDetector:
    def __init__(self, model_path="yolov8x.pt", confidence_thresh=0.5, suspicious_thresh=0.8):
        # Get project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # If model_path is relative, make it absolute
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        print(f"Loading model from: {model_path}")

        # Initialize YOLOv8 or RT-DETR model
        self.model = YOLO(model_path)
        self.conf = confidence_thresh
        self.suspicious_thresh = suspicious_thresh
        self.alert_system = AlertSystem()

        # Target classes: 0: tank, 1: military_vehicle, 2: drone, 3: ship, 4: soldier, 5: aircraft
        self.target_classes = [0, 1, 2, 3, 4, 5]
        self.suspicious_classes = {"tank", "military_vehicle", "drone", "ship", "soldier", "aircraft"}
        print(f"Detector initialized with {len(self.target_classes)} target classes")

    def process_stream(self, stream_url, camera_id):
        print(f"Opening stream: {stream_url}")
        handler = RTSPStreamHandler(stream_url)

        frame_count = 0
        while True:
            ret, frame = handler.read()
            if not ret:
                print(f"No more frames from {camera_id}")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count} from {camera_id}")

            # Perform tracking utilizing DeepSORT under the hood of Ultralytics tracking logic
            results = self.model.track(
                frame,
                conf=self.conf,
                classes=self.target_classes,
                persist=True,
                verbose=False,
            )

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = str(self.model.names[cls_id]).lower()

                    if cls_name in self.suspicious_classes and conf >= self.suspicious_thresh:
                        print(f"SUSPICIOUS ALERT: {cls_name} detected with {conf:.2f} confidence on {camera_id}")
                        self.alert_system.trigger_alert(cls_name, conf, camera_id)

        handler.release()
        print(f"Finished processing {frame_count} frames from {camera_id}")
