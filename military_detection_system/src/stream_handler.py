import cv2
import threading
import queue
import os


class RTSPStreamHandler:
    """
    Handles robust reading from video files or RTSP streams.
    """

    def __init__(self, stream_url, buffer_size=10):
        self.stream_url = stream_url

        # Check if it's a file path
        if os.path.isfile(stream_url):
            print(f"Opening video file: {stream_url}")
            self.cap = cv2.VideoCapture(stream_url)
            self.is_video_file = True
        else:
            print(f"Opening network stream: {stream_url}")
            self.cap = cv2.VideoCapture(stream_url)
            self.is_video_file = False

        if not self.cap.isOpened():
            print(f"ERROR: Could not open stream: {stream_url}")

        self.q = queue.Queue(maxsize=buffer_size)
        self.stopped = False

        # For video files, read in the same thread
        # For RTSP streams, use background thread
        if not self.is_video_file:
            self.t = threading.Thread(target=self._reader, daemon=True)
            self.t.start()

    def _reader(self):
        """Background reader for RTSP/HTTP streams."""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put(frame)
            else:
                # Drop oldest frame to maintain real-time queue
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

    def read(self):
        """Read a frame - for video files, read directly."""
        if self.is_video_file:
            ret, frame = self.cap.read()
            return ret, frame

        # For network streams, wait briefly for buffer to fill
        try:
            frame = self.q.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        self.stopped = True
        if hasattr(self, "t"):
            self.t.join(timeout=1.0)
        self.cap.release()
        print("Stream released")
