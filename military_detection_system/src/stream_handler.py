import cv2
import threading
import queue

class RTSPStreamHandler:
    """
    Handles robust reading from an RTSP stream in a background thread
    to ensure the main inference loop never stalls due to network delay.
    """
    def __init__(self, stream_url, buffer_size=10):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(self.stream_url)
        self.q = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
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
        if self.q.empty():
            return False, None
        return True, self.q.get()

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()
