import os
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "DS100-1-B2-Project-main" / "Project" / "dataset" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "DS100-1-B2-Project-main" / "Project" / "dataset" / "labels" / "train"

CLASS_NAMES = ["Apple", "Avocado", "Banana", "Carrot"] 
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

def draw_labels(image_path, label_path):
    img = cv2.imread(str(image_path))
    orig_h, orig_w, _ = img.shape

    img_resized = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
    scale_x = DISPLAY_WIDTH / orig_w
    scale_y = DISPLAY_HEIGHT / orig_h

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])

            x1 = int((x_center - box_width / 2) * orig_w * scale_x)
            y1 = int((y_center - box_height / 2) * orig_h * scale_y)
            x2 = int((x_center + box_width / 2) * orig_w * scale_x)
            y2 = int((y_center + box_height / 2) * orig_h * scale_y)

            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_resized, CLASS_NAMES[class_id], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_resized

for img_file in os.listdir(IMAGE_DIR):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = IMAGE_DIR / img_file
        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        if label_path.exists():
            labeled_img = draw_labels(img_path, label_path)
            cv2.imshow("Labeled Image", labeled_img)

            print(f"Showing: {img_file} (Press any key for next, Esc to exit)")

            key = cv2.waitKey(0)
            if key == 27:
                print("Exiting...")
                break
        else:
            print(f"⚠ No label found for {img_file}")

cv2.destroyAllWindows()
