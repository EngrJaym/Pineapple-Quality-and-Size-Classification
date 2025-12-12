"""
predict_folder.py
Run YOLOv8 OBB pineapple detection on all images inside a folder.

Just set:
    WEIGHTS_PATH
    IMAGE_FOLDER

Results will be saved to: runs/predict/
"""

import os
from ultralytics import YOLO

# -------------------------
# CONFIG
# -------------------------

WEIGHTS_PATH = "runs/obb/pineapple_obb/weights/best.pt"   # <-- CHANGE THIS
IMAGE_FOLDER = "images"                              # <-- CHANGE THIS

IMG_SIZE = 640
CONF_THRESHOLD = 0.65
DEVICE = "cpu"   # "0"=GPU0 | "cpu"=CPU

# -------------------------
# RUN PREDICTIONS
# -------------------------

def main():

    print(f"[INFO] Loading model: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    # Validate folder
    if not os.path.isdir(IMAGE_FOLDER):
        raise ValueError(f"[ERROR] Folder not found: {IMAGE_FOLDER}")

    print(f"[INFO] Running inference on all images in: {IMAGE_FOLDER}")

    # Run prediction on entire folder
    results = model.predict(
        source=IMAGE_FOLDER,   # <-- folder path
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        save=True,             # saves annotated results
        show=False             # change to True if you want preview windows
    )

    print("[INFO] Inference complete.")
    print("[INFO] Saved annotated images to: runs/predict/")

if __name__ == "__main__":
    main()
