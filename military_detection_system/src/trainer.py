from ultralytics import YOLO
import torch


def train_model():
    print("Loading base model...")

    # Use lightweight model (FAST + practical)
    model = YOLO("yolov8n.pt")

    print("Starting training on military dataset...")

    model.train(
        data="configs/training_config.yaml",
        # 🔥 OPTIMIZED SETTINGS
        epochs=30,  # reduced from 300
        imgsz=640,  # reduced from 1280
        batch=16,  # better utilization
        # Auto GPU detection
        device="0" if torch.cuda.is_available() else "cpu",
        name="military_v1",
        exist_ok=True,
        pretrained=True,
        # 🔥 EXTRA IMPROVEMENTS
        workers=2,  # stable for Colab
        patience=20,  # early stopping
        verbose=True,
    )

    print("Training complete.")


if __name__ == "__main__":
    train_model()
