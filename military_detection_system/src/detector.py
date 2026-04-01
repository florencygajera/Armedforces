"""
Military Detection System - Detector Module (UPDATED)
YOLOv8-based object detection for military surveillance
"""

import time
import cv2
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.alerts import AlertSystem
from src.stream_handler import RTSPStreamHandler


class MilitaryDetector:
    TARGET_CLASSES = {
        0: "tank",
        1: "military_vehicle",
        2: "drone",
        3: "ship",
        4: "soldier",
        5: "aircraft",
    }

    def __init__(
        self,
        model_path="runs/detect/military_v1/weights/best.pt",  # ✅ FIXED
        confidence_thresh=0.6,  # ✅ better threshold
    ):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        project_root = Path(__file__).parent.parent

        # Resolve model path
        if not Path(model_path).is_absolute():
            model_path = project_root / model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.logger.info(f"Loading model from: {model_path}")

        # Load trained model
        self.model = YOLO(str(model_path))

        self.confidence_thresh = confidence_thresh
        self.alert_system = AlertSystem()

        self.target_classes = list(self.TARGET_CLASSES.keys())

        self.frame_times = []
        self.detection_count = 0

        self.logger.info(
            f"Detector initialized with {len(self.target_classes)} classes"
        )

    def process_stream(
        self,
        stream_url,
        camera_id="cam_1",
        save_output=True,
        output_dir="logs/detections",
    ):
        self.logger.info(f"Opening stream: {stream_url}")

        handler = RTSPStreamHandler(stream_url)

        writer = None
        output_path = None

        if save_output:
            output_path = Path(output_dir) / f"{camera_id}_{int(time.time())}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = handler.read()
                if not ret:
                    break

                frame_count += 1
                frame_start = time.time()

                # 🔥 Detection + tracking
                results = self.model.track(
                    frame,
                    conf=self.confidence_thresh,
                    classes=self.target_classes,
                    persist=True,
                    verbose=False,
                )

                annotated_frame = self._process_results(results, frame, camera_id)

                # FPS calculation
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                avg_fps = (
                    1.0 / np.mean(self.frame_times[-30:]) if self.frame_times else 0
                )

                # Draw FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Initialize writer
                if save_output and writer is None:
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h))
                    self.logger.info(f"Saving output to: {output_path}")

                if writer:
                    writer.write(annotated_frame)

                if frame_count % 30 == 0:
                    self.logger.info(
                        f"Frame: {frame_count} | FPS: {avg_fps:.1f} | Detections: {self.detection_count}"
                    )

        finally:
            handler.release()
            if writer:
                writer.release()

            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0

            self.logger.info(
                f"Finished {frame_count} frames | Avg FPS: {avg_fps:.1f} | Total detections: {self.detection_count}"
            )

    def _process_results(self, results, frame, camera_id):
        annotated_frame = frame.copy()

        if not results:
            return annotated_frame

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 🔥 Filter target classes + confidence
                if cls_id in self.target_classes and conf > 0.8:
                    cls_name = self.TARGET_CLASSES.get(cls_id, f"class_{cls_id}")

                    # Trigger alert
                    self.alert_system.trigger_alert(cls_name, conf, camera_id)
                    self.detection_count += 1

                    self.logger.warning(
                        f"ALERT: {cls_name} detected ({conf:.2f}) on {camera_id}"
                    )

        return results[0].plot()

    def get_stats(self):
        if not self.frame_times:
            return {"fps": 0, "avg_frame_time": 0, "detections": 0}

        return {
            "fps": 1.0 / np.mean(self.frame_times[-30:]),
            "avg_frame_time": np.mean(self.frame_times),
            "detections": self.detection_count,
        }
