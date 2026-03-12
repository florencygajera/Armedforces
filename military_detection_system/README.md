# Real-Time Military Surveillance AI System

## PART 1 — SYSTEM ARCHITECTURE

The system employs a multi-stage pipeline designed for low-latency edge inference:

- **Frame Ingestion:** RTSP streams from CCTV, drone downlinks, or satellite feeds are captured in multi-threaded queues.
- **Inference Engine:** YOLOv8/RT-DETR accelerated by TensorRT performs object detection.
- **Object Tracking:** DeepSORT maintains consistent IDs across frames for trajectory analysis.
- **Alert System:** High-confidence detections of specific threats (e.g., tanks, drones) trigger MQTT/JSON logs.
- **Command Dashboard:** A Flask web interface visualizes incoming streams, tracks, and alerts.

## PART 2 — PROJECT STRUCTURE

```
military_detection_system/
├── venv/                 # Python virtual environment
├── data/                 # Raw data and assets
├── datasets/             # Prepared YOLO datasets (DOTA, xView)
├── models/               # Saved generic and custom weights
├── configs/              # System and training configurations (.yaml, .json)
├── src/                  # Core application logic (detector, tracking, alerts)
├── scripts/              # Helpers for dataset prep, conversion, model export
├── logs/                 # System logs and inference visualizations
├── runs/                 # YOLO training runs and checkpoints
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── Dockerfile            # GPU container definition
├── docker-compose.yml    # Multi-container orchestration (Detector, DB, MQTT)
└── README.md             # Project documentation
```

## PART 3 — HARDWARE REQUIREMENTS

- **Training (Min):** CPU: i5, RAM: 16GB, GPU: GTX 1060 6GB
- **Training (Rec):** CPU: i9, RAM: 32GB, GPU: RTX 3090 / A4000
- **Edge Deployment:** NVIDIA Jetson Orin/Xavier NX for optimal TensorRT throughput.

## PART 4 & 5 — SOFTWARE & SETUP

```bash
git clone https://github.com/example/military_detection_system
cd military_detection_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## PART 8 — MODEL TRAINING

```bash
yolo detect train model=yolov8x.pt data=configs/system_config.yaml epochs=300 imgsz=1280 batch=8
```

**Hyperparameters:** 

- `imgsz=1280`: Captures smaller details in aerial/CCTV footage.
- `epochs=300`: Full convergence requirement.
- `batch=8`: Balanced for large GPUs at high resolution.

## PART 9 — SMALL OBJECT OPTIMIZATION

To enhance detection of drones and soldiers:

1. **Image Tiling:** Slices 4K satellite images into small readable blocks.
2. **Multi-scale Training:** Randomly resizes input to make the network robust to distance.
3. **Focal Loss:** Penalizes easy examples to focus on difficult small objects.
4. **Higher Resolution (P2 Layer):** Extract features at stride 4.

## PART 13 — EDGE DEPLOYMENT

To deploy on Jetson Devices:

1. Export model to TensorRT engine using `scripts/export_model.py`.
2. Disable UI display if running headless.
3. Use FP16 inference to double throughput without losing accuracy.

