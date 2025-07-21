import os
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "DS100-1-B2-Project-main" / "Project" / "dataset" # Adjust this path as needed
IMAGE_DIRS = ["train", "val"]
CLASS_ID = 0  

def detect_fruit(image_path):
    img = cv2.imread(str(image_path))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 130])
    upper_white = np.array([180, 40, 240])
    lower_black = np.array([0, 0, 10])
    upper_black = np.array([180, 255, 90])

    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    mask_black = cv2.inRange(img_hsv, lower_black, upper_black)
    mask = cv2.bitwise_or(mask_white, mask_black)

    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = img.shape[:2]
    min_area = img_h * img_w * 0.005
    max_area = img_h * img_w * 0.5

    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    if not valid_contours:
        return None

    x, y, w, h = cv2.boundingRect(max(valid_contours, key=cv2.contourArea))
    expansion_factor = 0.1
    x = max(0, int(x - w * expansion_factor))
    y = max(0, int(y - h * expansion_factor))
    w = min(img_w - x, int(w * (1 + expansion_factor * 2)))
    h = min(img_h - y, int(h * (1 + expansion_factor * 2)))

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    return x_center, y_center, width, height

def create_yolo_labels():
    for image_dir in IMAGE_DIRS:
        image_folder = DATASET_PATH / "images" / image_dir # Adjust this path as needed
        label_folder = DATASET_PATH / "labels" / image_dir # Adjust this path as needed
        label_folder.mkdir(parents=True, exist_ok=True)

        for image_file in os.listdir(image_folder):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png",".txt")):
                image_path = image_folder / image_file
                label_path = label_folder / f"{image_path.stem}.txt"

                bbox = detect_fruit(image_path)
                if bbox is None:
                    print(f"Skipping {image_file} (No fruit detected)")
                    continue

                with open(label_path, "w") as f:
                    f.write(f"{CLASS_ID} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                print(f"Created label: {label_path}")

if __name__ == "__main__":
    create_yolo_labels()
