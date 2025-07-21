import torch
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "DS100-1-B2-Project-main" / "Project" / "dataset" / "data.yaml"

model = YOLO("yolov8s.pt")
model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=8,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
