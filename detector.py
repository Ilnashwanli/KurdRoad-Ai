from ultralytics import YOLO

MODEL_PATH = r"P:\Ai\KurdRoad-Ai\runs\pothole_v1\weights\best.pt"

model = YOLO(MODEL_PATH)


def detect_potholes(image_path):
    results = model.predict(
        source=image_path,
        conf=0.25,
        device=0
    )

    result = results[0]

    pothole_count = len(result.boxes)

    confidences = [
        float(box.conf[0])
        for box in result.boxes
    ]

    if confidences:
        average_confidence = sum(confidences) / len(confidences)
    else:
        average_confidence = 0

    if pothole_count == 0:
        severity = "No potholes detected"
    elif pothole_count <= 2:
        severity = "Low"
    elif pothole_count <= 5:
        severity = "Medium"
    else:
        severity = "High"

    return result, pothole_count, average_confidence, severity