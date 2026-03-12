from ultralytics import YOLO

def export_to_onnx(model_path="models/custom/best.pt"):
    model = YOLO(model_path)
    print("Exporting to ONNX...")
    model.export(format="onnx", dynamic=True, simplify=True)

def export_to_tensorrt(model_path="models/custom/best.pt"):
    model = YOLO(model_path)
    print("Exporting to TensorRT...")
    # INT8 or FP16 for Jetson devices
    model.export(format="engine", half=True, workspace=4)

if __name__ == "__main__":
    export_to_onnx("yolov8x.pt")
    # Tends to require Linux + CUDA tools installed natively
    export_to_tensorrt("yolov8x.pt")
