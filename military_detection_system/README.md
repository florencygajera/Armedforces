# Real-Time Military Surveillance AI System

A powerful AI-powered system designed for real-time military object detection and tracking using YOLOv8. The system processes video streams from various sources including CCTV cameras, drone feeds, and surveillance videos to detect and track military objects such as tanks, military vehicles, drones, ships, soldiers, and aircraft.

## Features

- **Real-Time Object Detection**: Uses YOLOv8 for fast and accurate detection
- **Multi-Stream Processing**: Process multiple camera streams simultaneously using threading
- **Object Tracking**: Maintains consistent IDs across frames using DeepSORT
- **Alert System**: High-confidence threat detection triggers alerts via MQTT/JSON
- **Web Dashboard**: Flask-based visualization interface for streams and alerts
- **Edge Deployment Ready**: Export models to TensorRT for NVIDIA Jetson devices

## Detection Classes

The system detects the following military objects:
- Tank
- Military Vehicle
- Drone
- Ship
- Soldier
- Aircraft

## Project Structure

```
military_detection_system/
├── configs/                 # Configuration files
│   ├── streams_config.json  # Camera/stream configurations
│   ├── system_config.yaml   # System settings
│   └── training_config.yaml # Training parameters
├── data/                    # Raw data and assets
├── datasets/               # YOLO format datasets
│   └── military/           # Military dataset
├── models/                 # Model weights storage
│   ├── custom/            # Custom trained models
│   └── pretrained/        # Pretrained models
├── scripts/                # Utility scripts
│   ├── convert_to_yolo.py  # Dataset conversion
│   ├── download_pretrained.py
│   ├── export_model.py    # TensorRT export
│   └── test_cameras.py    # Camera testing
├── src/                   # Core application
│   ├── alerts.py          # Alert system
│   ├── augmentations.py   # Image augmentations
│   ├── dashboard.py      # Web dashboard
│   ├── detector.py       # Detection logic
│   ├── main.py           # Entry point
│   ├── stream_handler.py # RTSP handling
│   ├── tracking.py       # Object tracking
│   ├── trainer.py        # Training logic
│   └── utils.py         # Utilities
├── logs/                 # System logs and outputs
├── runs/                 # Training runs and checkpoints
├── tests/                # Unit tests
├── videos/               # Video files
├── streams/              # Stream configurations
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container
└── docker-compose.yml    # Docker orchestration
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB RAM minimum
- GPU with 6GB+ VRAM

### Setup

```bash
# Clone the repository
git clone https://github.com/florencygajera/Armedforces.git
cd Armedforces/military_detection_system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Running the Detection System

```bash
# Configure your streams in configs/streams_config.json
# Then run:
python src/main.py
```

### Configuration

Edit `configs/streams_config.json` to add camera streams:

```json
{
    "cameras": [
        {
            "id": "camera_1",
            "type": "rtsp",
            "url": "rtsp://your-camera-url"
        },
        {
            "id": "video_file",
            "type": "video",
            "url": "videos/your-video.mp4"
        }
    ]
}
```

Edit `configs/system_config.yaml` to configure detection:

```yaml
model:
  path: "models/pretrained/yolov8x.pt"
  confidence_threshold: 0.5
  nms_threshold: 0.45

mqtt:
  broker: "localhost"
  port: 1883
  topic: "alerts/military"

display:
  show_video: false
  save_detections: true
```

## Model Training

### Training on Custom Dataset

```bash
yolo detect train model=yolov8x.pt data=configs/training_config.yaml epochs=300 imgsz=1280 batch=8
```

### Recommended Training Parameters

- `imgsz=1280`: Higher resolution for small object detection
- `epochs=300`: Full convergence
- `batch=8`: Balanced for GPUs with 16GB+ VRAM
- `mosaic=1.0`: Use mosaic augmentation for small object context

### Data Format

Place your dataset in YOLO format:
```
datasets/military/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Small Object Optimization

To enhance detection of small objects (drones, soldiers):

1. **Image Tiling**: Slice 4K satellite images into smaller blocks
2. **Multi-scale Training**: Random input resizing for distance robustness
3. **Focal Loss**: Focus on difficult examples
4. **Higher Resolution (P2 Layer)**: Extract features at stride 4

## Edge Deployment

### NVIDIA Jetson Setup

1. Export model to TensorRT:
```bash
python scripts/export_model.py
```

2. Run on Jetson Orin/Xavier NX:
```bash
# Disable display for headless mode
# Use FP16 inference for double throughput
```

### Docker Deployment

```bash
# Build container
docker build -t military-detection .

# Run with GPU
docker run --gpus all -v $(pwd):/app military-detection
```

Or use docker-compose:
```bash
docker-compose up --build
```

## Hardware Requirements

### Training
- **Minimum**: CPU i5, RAM 16GB, GPU GTX 1060 6GB
- **Recommended**: CPU i9, RAM 32GB, GPU RTX 3090/A4000

### Edge Deployment
- NVIDIA Jetson Orin
- NVIDIA Jetson Xavier NX

## Tech Stack

- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV
- **Tracking**: DeepSORT
- **MQTT**: Paho-MQTT
- **Web Dashboard**: Flask, Dash
- **Containerization**: Docker

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
