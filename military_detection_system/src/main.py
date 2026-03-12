"""
Military Detection System - Main Entry Point
Real-time object detection and tracking for military surveillance
"""

import sys
import os
import json
import threading
import time
from urllib.parse import urlparse
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import MilitaryDetector
from src.stream_handler import RTSPStreamHandler


def _is_stream_url(value):
    """Return True for network stream URLs (rtsp/http/https/rtmp)."""
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def setup_logging(level=logging.INFO):
    """Configure logging for the application"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/system.log')
        ]
    )


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def process_camera(camera_config, detector, save_output=False):
    """Process a single camera stream"""
    camera_id = camera_config['id']
    stream_url = camera_config['url']
    
    logging.info(f"Starting inference on {camera_id} ({stream_url})")
    
    try:
        detector.process_stream(stream_url, camera_id, save_output)
    except Exception as e:
        logging.error(f"Stream {camera_id} error: {e}")
        raise


def main():
    """Main entry point"""
    # Define project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description='Military Detection System')
    parser.add_argument('--config', '-c', default='configs/streams_config.json',
                        help='Path to streams configuration file')
    parser.add_argument('--model', '-m', default='yolov8x.pt',
                        help='Path to YOLO model')
    parser.add_argument('--confidence', '-conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save detection output to video')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Load streams configuration
    config_path = project_root / args.config
    logging.info(f"Loading streams from: {config_path}")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return
    
    cameras = config.get('cameras', [])
    
    if not cameras:
        logging.warning("No cameras configured")
        return
    
    # Process each camera
    for camera in cameras:
        # Resolve relative paths
        url = camera['url']
        if not os.path.isabs(url):
            camera['url'] = str(project_root / url)
    
    # Process first camera (can be extended to multi-threaded)
    if cameras:
        # Initialize detector
        detector = MilitaryDetector(
            model_path=args.model,
            confidence_thresh=args.confidence
        )
        process_camera(cameras[0], detector, args.save)


if __name__ == "__main__":
    main()
