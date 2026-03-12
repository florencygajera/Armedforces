import json
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

try:
    import winsound  # Windows only
except ImportError:  # pragma: no cover - non-Windows environments
    winsound = None


class AlertSystem:
    def __init__(
        self,
        mqtt_broker="localhost",
        mqtt_port=1883,
        topic="alerts/military",
        alert_cooldown_sec=5,
    ):
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            self.client = mqtt.Client()
        try:
            self.client.connect(mqtt_broker, mqtt_port, 60)
        except Exception as e:
            logging.warning(f"MQTT connection failed: {e}")

        self.topic = topic
        self.alert_cooldown_sec = alert_cooldown_sec
        self.last_alert_times = {}
        self._siren_lock = threading.Lock()

        self.logger = logging.getLogger("AlertSystem")
        self.logger.setLevel(logging.INFO)

        # Get project root and create logs directory path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Also log to a file
        handler = logging.FileHandler(os.path.join(logs_dir, "alerts.json"))
        self.logger.addHandler(handler)

    def _play_siren_windows(self, duration_sec):
        end_time = time.time() + duration_sec
        while time.time() < end_time:
            winsound.Beep(1300, 400)
            winsound.Beep(900, 400)

    def _play_siren_terminal(self, duration_sec):
        end_time = time.time() + duration_sec
        while time.time() < end_time:
            print("\a", end="", flush=True)
            time.sleep(0.25)

    def play_alert_sound(self, duration_sec=3):
        """Play a loud siren-like alert asynchronously."""

        def _run():
            with self._siren_lock:
                if platform.system().lower().startswith("win") and winsound is not None:
                    self._play_siren_windows(duration_sec)
                else:
                    self._play_siren_terminal(duration_sec)

        threading.Thread(target=_run, daemon=True).start()

    def _is_rate_limited(self, class_name, camera_id):
        key = f"{camera_id}:{class_name}"
        now = time.time()
        last_time = self.last_alert_times.get(key)
        if last_time is not None and now - last_time < self.alert_cooldown_sec:
            return True

        self.last_alert_times[key] = now
        return False

    def trigger_alert(self, class_name, confidence, camera_id):
        if self._is_rate_limited(class_name, camera_id):
            return None

        alert_payload = {
            "class": class_name,
            "confidence": round(confidence, 2),
            "camera_id": camera_id,
            "status": "suspicious_activity_detected",
            "message": f"Suspicious {class_name} detected on {camera_id}. Evacuate and investigate immediately.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Log as line-delimited JSON
        self.logger.info(json.dumps(alert_payload))

        # Play alert sound for nearby personnel
        self.play_alert_sound(duration_sec=4)

        # Send via MQTT for central command or public address integrations
        try:
            self.client.publish(self.topic, json.dumps(alert_payload))
        except Exception:
            pass

        return alert_payload
