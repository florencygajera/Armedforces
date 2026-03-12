from ultralytics import YOLO
import os
import sys
"""
Military Detection System - Detector Module
YOLOv8-based object detection for military surveillance
"""

import time
import cv2
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.alerts import AlertSystem
from src.stream_handler import StreamHandler



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
    """Main detector class for military object detection"""
    
    # COCO class names for reference
    TARGET_CLASSES = {
        0: 'tank',
        1: 'military_vehicle', 
        2: 'drone',
        3: 'ship',
        4: 'soldier',
        5: 'aircraft'
    }
    
    def __init__(self, model_path="yolov8x.pt", confidence_thresh=0.5):
        """Initialize the detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_thresh: Minimum confidence for detections
        """
        self.logger = logging.getLogger(__name__)
        
        # Get project root
        project_root = Path(__file__).parent.parent
        
        # Resolve model path
        if not Path(model_path).is_absolute():
            model_path = project_root / model_path
        
        self.logger.info(f"Loading model from: {model_path}")
        
        # Initialize YOLO model
        self.model = YOLO(str(model_path))
        self.confidence_thresh = confidence_thresh
        self.alert_system = AlertSystem()
        
        # Target classes for detection
        self.target_classes = list(self.TARGET_CLASSES.keys())
        
        # Performance tracking
        self.frame_times = []
        self.detection_count = 0
        
        self.logger.info(f"Detector initialized with {len(self.target_classes)} target classes")
    
    def process_stream(self, stream_url, camera_id, save_output=False, output_dir="logs/detections"):
        """Process a video stream
        
        Args:
            stream_url: URL or path to video stream
            camera_id: Identifier for the camera
            save_output: Whether to save annotated video
            output_dir: Directory to save output videos
        """
        self.logger.info(f"Opening stream: {stream_url}")
        
        handler = StreamHandler(stream_url)
        
        # Video writer for saving output
        writer = None
        if save_output:
            output_path = Path(output_dir) / f"{camera_id}_{int(time.time())}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = handler.read()
                if not ret:
                    self.logger.info(f"No more frames from {camera_id}")
                    break
                
                frame_count += 1
                frame_start = time.time()
                
                # Perform detection with tracking
                results = self.model.track(
                    frame, 
                    conf=self.confidence_thresh, 
                    classes=self.target_classes,
                    persist=True,
                    verbose=False
                )
                
                # Process detections
                annotated_frame = self._process_results(results, frame, camera_id)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                avg_fps = 1.0 / np.mean(self.frame_times[-30:]) if self.frame_times else 0
                
                # Add FPS display to frame
                cv2.putText(
                    annotated_frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Save output if enabled
                if save_output and writer is None:
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h))
                    self.logger.info(f"Saving output to: {output_path}")
                
                if writer:
                    writer.write(annotated_frame)
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    self.logger.info(
                        f"Processing frame {frame_count} | "
                        f"FPS: {avg_fps:.1f} | "
                        f"Detections: {self.detection_count}"
                    )
                
        finally:
            handler.release()
            if writer:
                writer.release()
            
            # Summary statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            self.logger.info(
                f"Finished processing {frame_count} frames from {camera_id} | "
                f"Avg FPS: {avg_fps:.1f} | "
                f"Total detections: {self.detection_count}"
            )
    
    def _process_results(self, results, frame, camera_id):
        """Process detection results
        
        Args:
            results: YOLO detection results
            frame: Original frame
            camera_id: Camera identifier
            
        Returns:
            Annotated frame with bounding boxes
        """
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Only alert for high confidence detections
                if conf > 0.8:
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    
                    # Trigger alert
                    self.alert_system.trigger_alert(cls_name, conf, camera_id)
                    self.detection_count += 1
                    
                    self.logger.warning(
                        f"ALERT: {cls_name} detected ({conf:.2f}) on {camera_id}"
                    )
        
        # Draw all detections on frame
        annotated_frame = results[0].plot() if results else annotated_frame
        
        return annotated_frame
    
    def get_stats(self):
        """Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.frame_times:
            return {'fps': 0, 'avg_frame_time': 0, 'detections': 0}
        
        return {
            'fps': 1.0 / np.mean(self.frame_times[-30:]),
            'avg_frame_time': np.mean(self.frame_times),
            'detections': self.detection_count
        }
