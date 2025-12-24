
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import shutil

# -------------------------
# CONFIG
# -------------------------

DETECTION_WEIGHTS_PATH = "detect.pt"  # <-- Detection model
CLASSIFICATION_WEIGHTS_PATH = "runs/classify/quality/weights/best.pt"  # <-- Classification model
IMAGE_FOLDER = "test images"  # <-- Input folder

DETECTION_IMG_SIZE = 640
CLASSIFICATION_IMG_SIZE = 224  # Must match training
DETECTION_CONF_THRESHOLD = 0.65  # Higher threshold to ensure it's actually a pineapple
if torch.cuda.is_available():
    DEVICE = "0"  # Use GPU
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"  # Fallback to CPU
    print("No GPU detected, using CPU")
# Output folders
ROTTEN_OUTPUT_FOLDER = "outputs/rotten"
FRESH_OUTPUT_FOLDER = "outputs/fresh"
ANNOTATED_OUTPUT_FOLDER = "runs/predict"


# -------------------------
# DETECTION FUNCTIONS
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
            'obb_points': obb_points
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


def read_image_with_fallback(img_path):
    """
    Try to read image with OpenCV first, fallback to PIL for AVIF and other formats.
    Returns None if unable to read the image.
    """
    # Try OpenCV first (handles most common formats)
    try:
        image = cv2.imread(img_path)
        if image is not None:
            return image
    except Exception as e:
        pass  # Continue to PIL fallback

    # Fallback to PIL for formats like AVIF, WEBP, etc.
    try:
        pil_image = Image.open(img_path)
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Convert PIL to OpenCV format (BGR)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        return None


# -------------------------
# CLASSIFICATION FUNCTIONS
# -------------------------

def classify_image(classifier_model, img_path):
    """
    Classify a full image as rotten or fresh using the correct method.
    Returns: (class_name, confidence) or (None, 0.0) if failed
    """
    try:
        results = classifier_model.predict(
            source=img_path,
            imgsz=CLASSIFICATION_IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        if len(results) == 0:
            return None, 0.0

        r = results[0]

        # Get prediction using the correct method
        class_id = int(r.probs.top1)
        confidence = float(r.probs.top1conf)
        class_name = classifier_model.names[class_id]

        return class_name, confidence

    except Exception as e:
        print(f"    ⚠️  Classification error: {e}")
        return None, 0.0


# -------------------------
# MAIN PIPELINE
# -------------------------

def main():
    # Create output directories
    os.makedirs(ROTTEN_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FRESH_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ANNOTATED_OUTPUT_FOLDER, exist_ok=True)

    print("=" * 70)
    print("PINEAPPLE DETECTION AND CLASSIFICATION PIPELINE")
    print("=" * 70)

    # Load models
    print(f"\n[STEP 1] Loading detection model: {DETECTION_WEIGHTS_PATH}")
    try:
        detection_model = YOLO(DETECTION_WEIGHTS_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load detection model: {e}")
        return

    print(f"[STEP 2] Loading classification model: {CLASSIFICATION_WEIGHTS_PATH}")
    try:
        classifier_model = YOLO(CLASSIFICATION_WEIGHTS_PATH)
        print(f"         Classification classes: {classifier_model.names}")
    except Exception as e:
        print(f"[ERROR] Could not load classification model: {e}")
        return

    # Validate folder
    if not os.path.isdir(IMAGE_FOLDER):
        raise ValueError(f"[ERROR] Folder not found: {IMAGE_FOLDER}")

    # Get all supported image files
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.avif')
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(supported_formats)
    ]

    if not image_files:
        print(f"[WARNING] No supported images found in {IMAGE_FOLDER}")
        return

    print(f"\n[STEP 3] Found {len(image_files)} images to process")
    print(f"         Detection image size: {DETECTION_IMG_SIZE}")
    print(f"         Classification image size: {CLASSIFICATION_IMG_SIZE}")
    print(f"         Detection confidence threshold: {DETECTION_CONF_THRESHOLD}")
    print(f"         Processing folder: {IMAGE_FOLDER}\n")
    print("-" * 70)

    # Statistics
    detection_count = 0
    rotten_count = 0
    fresh_count = 0
    no_detection_count = 0
    low_confidence_count = 0
    error_count = 0

    # Process each image
    for idx, img_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_file}")

        img_path = os.path.join(IMAGE_FOLDER, img_file)

        # Read image
        image = read_image_with_fallback(img_path)
        if image is None:
            print(f"    ❌ Could not read image (unsupported format or corrupted)")
            error_count += 1
            continue

        # ==================
        # DETECTION PHASE
        # ==================
        print(f"    🔍 Running detection...")
        try:
            results = detection_model.predict(
                source=img_path,
                imgsz=DETECTION_IMG_SIZE,
                conf=DETECTION_CONF_THRESHOLD,
                device=DEVICE,
                save=False,
                verbose=False
            )
        except Exception as e:
            print(f"    ❌ Detection failed: {e}")
            error_count += 1
            continue

        if len(results) == 0:
            print(f"    ❌ No pineapple detected (confidence < {DETECTION_CONF_THRESHOLD})")
            print(f"    ⏭️  Skipping classification")
            no_detection_count += 1
            continue

        # Get highest confidence box
        result = results[0]
        highest_box = get_highest_conf_box(result)

        if highest_box is None:
            print(f"    ❌ No pineapple detected (confidence < {DETECTION_CONF_THRESHOLD})")
            print(f"    ⏭️  Skipping classification")
            no_detection_count += 1
            continue

        # Check if confidence meets threshold
        if highest_box['conf'] < DETECTION_CONF_THRESHOLD:
            print(f"    ⚠️  Detection confidence too low: {highest_box['conf']:.3f} < {DETECTION_CONF_THRESHOLD}")
            print(f"    ⏭️  Skipping classification")
            low_confidence_count += 1
            continue

        detection_count += 1
        print(f"    ✓ Pineapple detected (confidence: {highest_box['conf']:.3f})")

        # ==================
        # CLASSIFICATION PHASE
        # ==================
        print(f"    🔬 Running classification on full image...")
        class_name, class_conf = classify_image(classifier_model, img_path)

        if class_name is None:
            print(f"    ❌ Classification failed")
            error_count += 1
            continue

        print(f"    ✓ Classification: {class_name.upper()} (confidence: {class_conf:.3f})")

        # Determine output folder based on classification
        if class_name.lower() == 'rotten':
            output_folder = ROTTEN_OUTPUT_FOLDER
            rotten_count += 1
            box_color = (0, 0, 255)  # Red (BGR format)
        else:  # fresh
            output_folder = FRESH_OUTPUT_FOLDER
            fresh_count += 1
            box_color = (0, 255, 0)  # Green (BGR format)

        # Copy original image to classification folder
        try:
            class_output_path = os.path.join(output_folder, img_file)
            shutil.copy2(img_path, class_output_path)
            print(f"    💾 Classified image saved: {class_output_path}")
        except Exception as e:
            print(f"    ⚠️  Could not save classified image: {e}")

        # Create annotated image with classification
        try:
            annotated_img = result.orig_img.copy()

            # Draw bounding box with classification color
            if highest_box['obb_points'] is not None:
                points = highest_box['obb_points'].astype(np.int32)
                cv2.polylines(annotated_img, [points], True, box_color, 3)
                label_pos = (int(points[0][0]), int(points[0][1]) - 10)
            else:
                x1, y1, x2, y2 = map(int, highest_box['box'])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
                label_pos = (x1, y1 - 10)

            # Add classification label with background for better visibility
            label = f"{class_name.upper()}: {class_conf:.2f}"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            # Draw background rectangle
            cv2.rectangle(
                annotated_img,
                (label_pos[0], label_pos[1] - text_height - baseline - 5),
                (label_pos[0] + text_width, label_pos[1] + baseline),
                box_color,
                -1  # Filled
            )

            # Draw text
            cv2.putText(
                annotated_img,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # White text
                2
            )

            # Save annotated image
            annotated_path = os.path.join(ANNOTATED_OUTPUT_FOLDER, img_file)
            cv2.imwrite(annotated_path, annotated_img)
            print(f"    💾 Annotated image saved: {annotated_path}")

        except Exception as e:
            print(f"    ⚠️  Could not save annotated image: {e}")

    # ==================
    # SUMMARY
    # ==================
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Total images processed:     {len(image_files)}")
    print(f"Pineapples detected:        {detection_count}")
    print(f"No pineapple detected:      {no_detection_count}")
    print(f"Low confidence detections:  {low_confidence_count}")
    print(f"Errors/Unsupported:         {error_count}")
    print(f"\nClassification Results (only images with detected pineapples):")
    print(f"  🟢 Fresh:                 {fresh_count}")
    print(f"  🔴 Rotten:                {rotten_count}")
    print(f"\nOutput Locations:")
    print(f"  📁 Annotated images:      {ANNOTATED_OUTPUT_FOLDER}/")
    print(f"  📁 Fresh pineapples:      {FRESH_OUTPUT_FOLDER}/")
    print(f"  📁 Rotten pineapples:     {ROTTEN_OUTPUT_FOLDER}/")
    print("=" * 70)


if __name__ == "__main__":
    main()