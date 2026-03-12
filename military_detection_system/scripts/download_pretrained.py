import urllib.request
import os

def download_yolov8x():
    url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt"
    dest = "models/pretrained/yolov8x.pt"
    print(f"Downloading {url} to {dest}...")
    # urllib.request.urlretrieve(url, dest)
    print("For production, recommend downloading models manually or utilizing Ultralytics library auto-download.")

if __name__ == "__main__":
    download_yolov8x()
