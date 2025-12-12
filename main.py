"""
train_yolov8_obb.py
Training YOLOv8 Oriented Bounding Box model (OBB) for detecting pineapples.
Dataset must be exported in YOLOv8 OBB format from Roboflow.

Run:
    python train_yolov8_obb.py
"""

import os
from ultralytics import YOLO

# -------------------------
# CONFIGURATION
# -------------------------

DATASET_PATH = "dataset"   # <-- CHANGE THIS
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")

# Only 1 class
NUM_CLASSES = 1
CLASS_NAMES = ["pineapple"]

# Real-time model
MODEL_NAME = "yolov8n-obb.pt"

# Hyperparameters
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
WORKERS = 4
DEVICE = "cpu"

# -------------------------
# TRAINING
# -------------------------

def main():

    print("[INFO] Loading OBB model:", MODEL_NAME)
    model = YOLO(MODEL_NAME)

    print("[INFO] Training YOLOv8 OBB model...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        optimizer="AdamW",

        # Early stopping here
        patience=20,            # 🔥 stops if no improvement for 20 epochs

        # Strong augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        project="runs/obb",
        name="pineapple_obb",
        exist_ok=True
    )

    print("[INFO] Training complete.")

    best_model_path = "runs/obb/pineapple_obb/weights/best.pt"
    print(f"[INFO] Best model saved at: {best_model_path}")

    print("[INFO] Exporting ONNX model...")
    model.export(format="onnx")

    print("[INFO] OBB training pipeline completed.")

if __name__ == "__main__":
    main()
