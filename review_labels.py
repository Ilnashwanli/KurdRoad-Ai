from pathlib import Path
from PIL import Image, ImageDraw

# ==============================
# PATHS
# ==============================

IMAGE_DIR = Path(r"P:\Ai\KurdRoad-Ai\local_data\raw\potholes")

LABEL_DIR = Path(r"P:\Ai\KurdRoad-Ai\local_data\raw\potholes_labels")

OUTPUT_DIR = Path(r"P:\Ai\KurdRoad-Ai\local_data\raw\reviewed_potholes")

# Create output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# FIND IMAGES
# ==============================

extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
]

print(f"Found {len(images)} images.")
print()

# ==============================
# DRAW LABELS
# ==============================

for i, image_path in enumerate(images, start=1):

    print(f"[{i}/{len(images)}] {image_path.name}")

    # Open image
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    # Find corresponding label
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    box_count = 0

    if label_path.exists():

        with open(label_path, "r") as f:

            lines = f.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id, x_center, y_center, box_width, box_height = map(
                float, parts
            )

            # Convert YOLO normalized coordinates
            # to image coordinates

            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw box
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=4
            )

            # Draw label
            draw.text(
                (x1, max(0, y1 - 20)),
                "POTHOLE",
                fill="red"
            )

            box_count += 1

    # Add information at top
    draw.text(
        (10, 10),
        f"Potholes detected: {box_count}",
        fill="yellow"
    )

    # Save reviewed image
    output_path = OUTPUT_DIR / image_path.name

    image.save(output_path)

    print(f"    Boxes: {box_count}")
    print(f"    Saved: {output_path}")

print()
print("====================================")
print("VISUAL REVIEW IMAGES CREATED!")
print("====================================")
print()
print(f"Open this folder:")
print(OUTPUT_DIR)