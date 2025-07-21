import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
IMAGE_DIR = PROJECT_ROOT / "DS100-1-B2-Project-main" / "Project" / "test_images"
RESULTS_CSV = PROJECT_ROOT / "fruit_identification.csv"

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
CLASS_NAMES = ["Apple", "Avocado", "Banana", "Carrot"]

model = YOLO(str(MODEL_PATH))

def process_images():
    results_data = []

    for img_file in os.listdir(IMAGE_DIR):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png", "txt")):
            img_path = IMAGE_DIR / img_file
            img = cv2.imread(str(img_path))

            results = model(img)

            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    label = CLASS_NAMES[class_id]
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detections.append({"label": label, "confidence": confidence, "bbox": (x1, y1, x2, y2)})

            print(f"Image: {img_file} - Detected Objects: {detections}")
            results_data.append({"Image": img_file, "Detections": len(detections)})

            draw_boxes(img, detections, img_file)

    df = pd.DataFrame(results_data)
    df.to_csv(RESULTS_CSV, index=False)

    return df

def draw_boxes(img, detections, img_name):
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

    scale_x = DISPLAY_WIDTH / orig_w
    scale_y = DISPLAY_HEIGHT / orig_h

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_resized, f"{det['label']} ({confidence:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection", img_resized)
    print(f"Showing: {img_name} (Press any key for next, Esc to exit)")
    key = cv2.waitKey(0)

    if key == 27:
        print("Exiting...")
        cv2.destroyAllWindows()
        exit()

    cv2.destroyAllWindows()

def plot_detections(df):
    plt.figure(figsize=(10, 5))
    plt.bar(df["Image"], df["Detections"], color="b")
    plt.xlabel("Image")
    plt.ylabel("Number of Detections")
    plt.title("Fuit Identification Results")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    df = process_images()
    plot_detections(df)
