from ultralytics import YOLO
import torch

DATASET_PATH = "quality_classification"
MODEL_NAME = "yolov8m-cls.pt"   # use SMALL, nano is too weak for texture learning

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_NAME)

model.train(
    data=DATASET_PATH,
    epochs=80,
    imgsz=224,
    batch=16,                  # smaller batch = better generalization
    device=DEVICE,

    optimizer="AdamW",
    lr0=3e-4,
    lrf=1e-2,
    weight_decay=5e-4,

    label_smoothing=0.1,

    # REALISTIC augmentations ONLY
    hsv_h=0.01,
    hsv_s=0.15,
    hsv_v=0.15,
    degrees=7,
    translate=0.05,
    scale=0.05,
    fliplr=0.5,
    flipud=0.0,

    # Disable harmful aug
    erasing=0.0,
    auto_augment=None,

    patience=20,

    project="runs/classify",
    name="quality",
    exist_ok=True
)
