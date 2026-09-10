from pathlib import Path
import shutil
import random

# ==========================================
# PATHS
# ==========================================

ROOT = Path(r"P:\Ai\KurdRoad-Ai")

POTHOLE_DIR = ROOT / "local_data" / "raw" / "potholes"
POTHOLE_LABEL_DIR = ROOT / "local_data" / "raw" / "potholes_labels"

GOOD_DIR = ROOT / "local_data" / "raw" / "good_roads"

DATASET = ROOT / "dataset"

# ==========================================
# SETTINGS
# ==========================================

random.seed(42)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==========================================
# CREATE FOLDERS
# ==========================================

for split in ["train", "val", "test"]:
    (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# ==========================================
# GET IMAGES
# ==========================================

potholes = sorted([
    p for p in POTHOLE_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
])

good_roads = sorted([
    p for p in GOOD_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
])

print(f"Pothole images: {len(potholes)}")
print(f"Good road images: {len(good_roads)}")

# ==========================================
# SHUFFLE
# ==========================================

random.shuffle(potholes)
random.shuffle(good_roads)

# ==========================================
# SPLIT
# ==========================================

pothole_train = potholes[:20]
pothole_val = potholes[20:24]
pothole_test = potholes[24:28]

good_train = good_roads[:6]
good_val = good_roads[6:7]
good_test = good_roads[7:8]

splits = {
    "train": pothole_train + good_train,
    "val": pothole_val + good_val,
    "test": pothole_test + good_test,
}

# ==========================================
# COPY IMAGES + LABELS
# ==========================================

for split, images in splits.items():

    print()
    print(f"Preparing {split}: {len(images)} images")

    for image_path in images:

        # Copy image
        destination_image = DATASET / "images" / split / image_path.name
        shutil.copy2(image_path, destination_image)

        # --------------------------------------
        # POTHOLE IMAGE
        # --------------------------------------

        if image_path.parent == POTHOLE_DIR:

            source_label = POTHOLE_LABEL_DIR / f"{image_path.stem}.txt"
            destination_label = DATASET / "labels" / split / f"{image_path.stem}.txt"

            if source_label.exists():
                shutil.copy2(source_label, destination_label)
            else:
                print(f"WARNING: Missing label: {source_label}")

        # --------------------------------------
        # NORMAL ROAD
        # --------------------------------------

        else:

            # Empty label = no potholes
            destination_label = DATASET / "labels" / split / f"{image_path.stem}.txt"

            destination_label.write_text("")

print()
print("==========================================")
print("DATASET PREPARATION FINISHED")
print("==========================================")
print()

for split in ["train", "val", "test"]:

    image_count = len(list((DATASET / "images" / split).iterdir()))
    label_count = len(list((DATASET / "labels" / split).iterdir()))

    print(f"{split}: {image_count} images / {label_count} labels")

print()
print(f"Dataset location: {DATASET}")