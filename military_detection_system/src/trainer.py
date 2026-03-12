from ultralytics import YOLO

def train_model():
    print("Loading base model...")
    model = YOLO('yolov8x.pt')
    
    # Train the model with focal loss implicitly balanced by class weights
    # and utilize high-res P2 layers natively by passing imgsz=1280
    print("Starting training on military dataset...")
    results = model.train(
        data='configs/training_config.yaml',
        epochs=300,
        imgsz=1280,
        batch=8,
        device='0' if __import__('torch').cuda.is_available() else 'cpu', # Auto-detect GPU or CPU
        name='military_v1',
        exist_ok=True,
        pretrained=True
    )
    
    print("Training complete.")

if __name__ == "__main__":
    train_model()
