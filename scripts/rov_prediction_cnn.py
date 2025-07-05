from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (e.g., YOLOv8n - nano version)
model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data="/home/costin/Downloads/SeaClear ROV.v1-seaclear-training-dataset.yolov8/data.yaml",  # path to your data.yaml
    epochs=175,
    imgsz=640,
    batch=16,
    device=0  # set to 'cpu' if no GPU available
)

# Save the model
model_path = model.ckpt_path if hasattr(model, "ckpt_path") else "runs/detect/train/weights/best.pt"
print(f"Model trained and saved at: {model_path}")