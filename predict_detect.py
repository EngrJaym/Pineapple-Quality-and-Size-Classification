"""
predict_folder.py
Run YOLOv8 OBB pineapple detection on all images inside a folder.
- Only keeps highest confidence detection per image
- Saves cropped detections to outputs/crops/
- Supports AVIF and WEBP formats

Just set:
    WEIGHTS_PATH
    IMAGE_FOLDER

Results will be saved to: runs/predict/ (annotated images)
                          outputs/crops/ (cropped detections)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# -------------------------
# CONFIG
# -------------------------

WEIGHTS_PATH = "detect.pt"   # <-- CHANGE THIS
IMAGE_FOLDER = "test images"  # <-- CHANGE THIS

IMG_SIZE = 640
CONF_THRESHOLD = 0.65
DEVICE = "cpu"   # "0"=GPU0 | "cpu"=CPU

# Output folder for crops
CROP_OUTPUT_FOLDER = "outputs/crops"

# -------------------------
# RUN PREDICTIONS
# -------------------------

def get_highest_conf_box(result):
    """
    Extract only the highest confidence bounding box from a result.
    Handles both regular boxes and OBB (Oriented Bounding Boxes).
    Returns None if no detections.
    """
    # Check for OBB first
    if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
        confidences = result.obb.conf.cpu().numpy()
        max_idx = np.argmax(confidences)

        # Get OBB xyxyxyxy format (4 corner points)
        obb_points = result.obb.xyxyxyxy[max_idx].cpu().numpy()

        # Convert OBB to regular bounding box (min/max of all points)
        x_coords = obb_points[:, 0]
        y_coords = obb_points[:, 1]
        x1, y1 = x_coords.min(), y_coords.min()
        x2, y2 = x_coords.max(), y_coords.max()

        return {
            'box': np.array([x1, y1, x2, y2]),
            'conf': confidences[max_idx],
            'cls': result.obb.cls[max_idx].cpu().numpy(),
            'obb_points': obb_points  # Keep original OBB for better visualization
        }

    # Check for regular boxes
    elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        confidences = result.boxes.conf.cpu().numpy()
        max_idx = np.argmax(confidences)

        return {
            'box': result.boxes.xyxy[max_idx].cpu().numpy(),
            'conf': confidences[max_idx],
            'cls': result.boxes.cls[max_idx].cpu().numpy(),
            'obb_points': None
        }

    return None

def save_crop(image, box_info, output_path):
    """
    Save cropped region from image based on bounding box.
    """
    x1, y1, x2, y2 = map(int, box_info['box'])

    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Crop and save
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, crop)

def read_image_with_fallback(img_path):
    """
    Try to read image with OpenCV first, fallback to PIL for AVIF.
    """
    # Try OpenCV first
    image = cv2.imread(img_path)
    if image is not None:
        return image

    # Fallback to PIL for formats like AVIF
    try:
        pil_image = Image.open(img_path)
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Convert PIL to OpenCV format (BGR)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        print(f"[ERROR] Could not read {img_path}: {e}")
        return None

def main():
    # Create output directory
    os.makedirs(CROP_OUTPUT_FOLDER, exist_ok=True)

    print(f"[INFO] Loading model: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    # Validate folder
    if not os.path.isdir(IMAGE_FOLDER):
        raise ValueError(f"[ERROR] Folder not found: {IMAGE_FOLDER}")

    # Get all supported image files (including AVIF and WEBP)
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.avif')
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(supported_formats)
    ]

    if not image_files:
        print(f"[WARNING] No supported images found in {IMAGE_FOLDER}")
        return

    print(f"[INFO] Found {len(image_files)} images (including AVIF/WEBP)")
    print(f"[INFO] Running inference on: {IMAGE_FOLDER}")
    print(f"[INFO] Confidence threshold: {CONF_THRESHOLD}\n")

    detection_count = 0

    # Process each image individually
    for img_file in image_files:
        img_path = os.path.join(IMAGE_FOLDER, img_file)

        # Read image with fallback for AVIF
        image = read_image_with_fallback(img_path)
        if image is None:
            print(f"[WARNING] Could not read: {img_file}")
            continue

        # Run prediction
        results = model.predict(
            source=img_path,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            save=False,
            verbose=False  # Suppress per-image output
        )

        if len(results) == 0:
            print(f"[INFO] {img_file}: No detections")
            continue

        # Get highest confidence box only
        result = results[0]
        highest_box = get_highest_conf_box(result)

        if highest_box is None:
            print(f"[INFO] {img_file}: No detections")
            continue

        detection_count += 1
        print(f"[INFO] {img_file}: Detection (conf={highest_box['conf']:.3f})")

        # Save annotated image with only the highest confidence box
        annotated_img = result.orig_img.copy()

        # Draw OBB if available, otherwise regular box
        if highest_box['obb_points'] is not None:
            points = highest_box['obb_points'].astype(np.int32)
            cv2.polylines(annotated_img, [points], True, (0, 255, 0), 2)
            label_pos = tuple(points[0])
        else:
            x1, y1, x2, y2 = map(int, highest_box['box'])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_pos = (x1, y1 - 10)

        # Add confidence label
        label = f"{highest_box['conf']:.2f}"
        cv2.putText(annotated_img, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save annotated image
        annotated_path = os.path.join("runs/predict", img_file)
        os.makedirs("runs/predict", exist_ok=True)
        cv2.imwrite(annotated_path, annotated_img)

        # Save crop
        crop_filename = f"{Path(img_file).stem}_crop{Path(img_file).suffix}"
        crop_path = os.path.join(CROP_OUTPUT_FOLDER, crop_filename)
        save_crop(image, highest_box, crop_path)

        print(f"       → Crop saved to: {crop_path}")

    print(f"\n[INFO] Inference complete.")
    print(f"[INFO] Processed {len(image_files)} images, found {detection_count} detections")
    print(f"[INFO] Annotated images saved to: runs/predict/")
    print(f"[INFO] Cropped detections saved to: {CROP_OUTPUT_FOLDER}/")

if __name__ == "__main__":
    main()