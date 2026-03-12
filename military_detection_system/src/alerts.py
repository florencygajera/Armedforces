import json
import logging
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
        
        # Also log to a file
        handler = logging.FileHandler('logs/alerts.json')
        self.logger.addHandler(handler)

    def trigger_alert(self, class_name, confidence, camera_id):
        alert_payload = {
            "class": class_name,
            "confidence": round(confidence, 2),
            "camera_id": camera_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Log to JSON
        self.logger.info(json.dumps(alert_payload))
        
        # Send via MQTT
        try:
            self.client.publish(self.topic, json.dumps(alert_payload))
        except Exception as e:
            pass
            
        return alert_payload
