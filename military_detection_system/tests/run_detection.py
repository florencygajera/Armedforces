from src.detector import MilitaryDetector

detector = MilitaryDetector()

detector.process_stream(
    stream_url="videos/test.mp4",  # change this
    camera_id="cam_1",
    save_output=True,
)
