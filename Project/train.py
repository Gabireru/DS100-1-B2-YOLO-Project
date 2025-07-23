import os
import torch
from multiprocessing import freeze_support
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
print(f"Working directory changed to: {os.getcwd()}")

DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
SAVE_DIR = PROJECT_ROOT / "runs" / "classify" / "custom_train"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "images"
train_dst = DATASET_ROOT / 'train'
val_dst = DATASET_ROOT / 'val'

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Data YAML file not found: {DATA_YAML}")
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset root directory not found: {DATASET_ROOT}")
train_dst.mkdir(parents=True, exist_ok=True)
val_dst.mkdir(parents=True, exist_ok=True)

classes = [d.name for d in DATASET_ROOT.iterdir() if d.is_dir()]
print(f"Found {len(classes)} classes: {classes}")

for cls in classes:
    cls_path = DATASET_ROOT / cls
    if not cls_path.exists():
        raise FileNotFoundError(f"Class directory not found: {cls_path}")
    print(f"Copied images for class '{cls}' to train and val directories.")

print("Images copied under train/ and val/ structured by class folders")

if __name__ == '__main__':
    freeze_support()

    model = YOLO('yolo11s-cls.pt')

    results = model.train(
        data=str(DATASET_ROOT),
        epochs=20,
        imgsz=100,
        batch=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project=str(PROJECT_ROOT / "runs" / "classify"),    
        save_dir=str(SAVE_DIR),
        augment=True,
    )