from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Command Center Dashboard</h1><p>API is running. Connect via specific endpoints.</p>"

@app.route('/api/alerts/recent', methods=['GET'])
def get_recent_alerts():
    alerts = []
    if os.path.exists("logs/alerts.json"):
        with open("logs/alerts.json", "r") as f:
            for line in f.readlines()[-10:]: # last 10 alerts
                try:
                    alerts.append(json.loads(line.strip()))
                except Exception:
                    pass
    return jsonify({"alerts": alerts})

if __name__ == "__main__":
    # In production use Waitress or Gunicorn
    app.run(host="0.0.0.0", port=5000)
