import os
from ultralytics import YOLO

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "runs/classify/quality/weights/best.pt"        # path to your YOLOv8-cls model
IMAGE_FOLDER = "test images" # folder with images
DEVICE = "cpu"               # "cuda" if GPU works
IMG_SIZE = 224               # must match training

# -------------------------
# MAIN
# -------------------------
def main():
    # Load model
    print("[INFO] Loading classification model...")
    model = YOLO(MODEL_PATH)

    # Validate image folder
    if not os.path.isdir(IMAGE_FOLDER):
        raise ValueError(f"[ERROR] Folder not found: {IMAGE_FOLDER}")

    # Get images
    images = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    if not images:
        raise ValueError("[ERROR] No images found in folder.")

    print(f"[INFO] Found {len(images)} images\n")

    # Run prediction per image
    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        results = model.predict(
            source=img_path,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        r = results[0]

        # Get prediction
        class_id = int(r.probs.top1)
        confidence = float(r.probs.top1conf)
        class_name = model.names[class_id]

        print(f"{img_name} → {class_name.upper()} ({confidence:.2f})")

    print("\n[INFO] Prediction complete.")

# -------------------------
if __name__ == "__main__":
    main()
