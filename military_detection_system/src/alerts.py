import json
import logging
import os
import sys
import winsound
from datetime import datetime
import paho.mqtt.client as mqtt

class AlertSystem:
    def __init__(self, mqtt_broker="localhost", mqtt_port=1883, topic="alerts/military"):
        self.client = mqtt.Client()
        try:
            self.client.connect(mqtt_broker, mqtt_port, 60)
        except Exception as e:
            logging.warning(f"MQTT connection failed: {e}")
            
        self.topic = topic
        self.logger = logging.getLogger("AlertSystem")
        self.logger.setLevel(logging.INFO)
        
        # Get project root and create logs directory path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Also log to a file
        handler = logging.FileHandler(os.path.join(logs_dir, 'alerts.json'))
        self.logger.addHandler(handler)

    def play_alert_sound(self):
        """Play a system beep sound for alert"""
        try:
            # Play Windows system sound (beep)
            winsound.PlaySound(winsound.ALERT, winsound.SND_ASYNC)
        except Exception as e:
            print(f"Could not play sound: {e}")

    def trigger_alert(self, class_name, confidence, camera_id):
        alert_payload = {
            "class": class_name,
            "confidence": round(confidence, 2),
            "camera_id": camera_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Log to JSON
        self.logger.info(json.dumps(alert_payload))
        
        # Play alert sound
        print(f"\a")  # Terminal beep
        self.play_alert_sound()
        
        # Send via MQTT
        try:
            self.client.publish(self.topic, json.dumps(alert_payload))
        except Exception as e:
            pass
            
        return alert_payload
