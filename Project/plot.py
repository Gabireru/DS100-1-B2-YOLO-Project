import os
import shutil
from pathlib import Path
from ultralytics import YOLO # YOLO is not strictly needed if only plotting
import matplotlib.pyplot as plt
import pandas as pd # Import pandas for CSV reading

# Define the path to your results.csv file.
# You'll need to adjust this to where your 'results.csv' is actually located.
# For example, if it's in a 'runs/classify/custom_train' directory:
PROJECT_ROOT = Path(__file__).resolve().parent
# Assuming results.csv is within your SAVE_DIR structure
SAVE_DIR = PROJECT_ROOT / "runs" / "classify" / "custom_train(20)"
results_csv_path = SAVE_DIR / "results.csv"

# Check if the results.csv file exists
if not results_csv_path.exists():
    raise FileNotFoundError(f"results.csv not found at: {results_csv_path}")

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(results_csv_path)

    # Extract accuracy and loss from the DataFrame
    # Note: Ultralytics often names these columns specifically.
    # Common column names for classification metrics are:
    # 'metrics/accuracy_top1', 'train/loss', 'val/loss'
    # You might need to inspect your 'results.csv' to confirm the exact column names.

    # Try common Ultralytics column names for classification
    accuracy_col = 'metrics/accuracy_top1'
    loss_col = 'train/loss' # Or 'val/loss' depending on what you want to plot

    # Check if the columns exist, otherwise try alternatives or print a warning
    if accuracy_col not in df.columns:
        print(f"Warning: '{accuracy_col}' not found. Trying 'val/accuracy_top1'.")
        accuracy_col = 'val/accuracy_top1'
        if accuracy_col not in df.columns:
            raise ValueError(f"Neither 'metrics/accuracy_top1' nor 'val/accuracy_top1' found in {results_csv_path}. Please check column names.")

    if loss_col not in df.columns:
        print(f"Warning: '{loss_col}' not found. Trying 'val/loss'.")
        loss_col = 'val/loss'
        if loss_col not in df.columns:
            raise ValueError(f"Neither 'train/loss' nor 'val/loss' found in {results_csv_path}. Please check column names.")

    accuracy = df[accuracy_col].dropna().tolist()
    loss = df[loss_col].dropna().tolist()

    # Ensure that accuracy and loss lists are not empty
    if not accuracy or not loss:
        raise ValueError("Accuracy or Loss data is empty after reading CSV. Check your CSV content.")

    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, marker='o', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, marker='x', color='red', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # Save the plot to the same directory as the CSV, or your SAVE_DIR
    plt.savefig(SAVE_DIR / "training_metrics_from_csv.png")
    print(f"Training metrics plot saved to: {SAVE_DIR / 'training_metrics_from_csv.png'}")
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Please ensure '{results_csv_path}' exists and has the correct columns for accuracy and loss.")
    print("Common Ultralytics classification columns are 'metrics/accuracy_top1', 'train/loss', 'val/accuracy_top1', 'val/loss'.")