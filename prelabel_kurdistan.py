from ultralytics import YOLO
from pathlib import Path

# ==============================
# PATHS
# ==============================

MODEL_PATH = r"P:\Ai\KurdRoad-Ai\runs\pothole_v1\weights\best.pt"

IMAGE_DIR = Path(r"P:\Ai\KurdRoad-Ai\local_data\raw\potholes")

LABEL_DIR = Path(r"P:\Ai\KurdRoad-Ai\local_data\raw\potholes_labels")

# ==============================
# SETTINGS
# ==============================

CONFIDENCE = 0.15
DEVICE = 0

# ==============================
# LOAD MODEL
# ==============================

print("Loading YOLOv8 model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print()

# Create label folder
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Find images
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

print(f"Found {len(images)} images.")
print()

# ==============================
# PRE-LABEL IMAGES
# ==============================

for i, image_path in enumerate(images, start=1):

    print(f"[{i}/{len(images)}] Processing: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    # YOLO label file
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with open(label_path, "w") as f:

        if result.boxes is not None:

            for box in result.boxes:

                # Class ID
                class_id = int(box.cls[0])

                # Normalized YOLO format:
                # x_center y_center width height
                x, y, w, h = box.xywhn[0].tolist()

                f.write(
                    f"{class_id} "
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}\n"
                )

    print(f"    Saved: {label_path.name}")

print()
print("====================================")
print("PRE-LABELING FINISHED!")
print("====================================")
print()
print(f"Images: {IMAGE_DIR}")
print(f"Labels: {LABEL_DIR}")