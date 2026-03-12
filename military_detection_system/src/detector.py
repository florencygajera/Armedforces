from ultralytics import YOLO
import cv2
from src.alerts import AlertSystem
from src.stream_handler import RTSPStreamHandler

class MilitaryDetector:
    def __init__(self, model_path="yolov8x.pt", confidence_thresh=0.5):
        # Initialize YOLOv8 or RT-DETR model
        self.model = YOLO(model_path)
        self.conf = confidence_thresh
        self.alert_system = AlertSystem()
        
        # Target classes: 0: tank, 1: military_vehicle, 2: drone, 3: ship, 4: soldier, 5: aircraft
        self.target_classes = [0, 1, 2, 3, 4, 5] 

    def process_stream(self, stream_url, camera_id):
        handler = RTSPStreamHandler(stream_url)
        
        while True:
            ret, frame = handler.read()
            if not ret:
                continue
                
            # Perform tracking utilizing DeepSORT under the hood of Ultralytics tracking logic
            results = self.model.track(frame, conf=self.conf, classes=self.target_classes, persist=True, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = self.model.names[cls_id]
                    
                    if conf > 0.8: # High confidence threat alert
                        self.alert_system.trigger_alert(cls_name, conf, camera_id)
                        
            # Optionally visualize
            annotated_frame = results[0].plot()
            cv2.imshow(camera_id, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        handler.release()
