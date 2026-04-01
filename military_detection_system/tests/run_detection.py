import os
import sys

# Ensure the project root is in sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import MilitaryDetector

detector = MilitaryDetector()

detector.process_stream(
    stream_url="videos/test.mp4",  # change this
    camera_id="cam_1",
    save_output=True,
)
