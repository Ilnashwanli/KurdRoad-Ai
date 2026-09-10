# 🛣️ KurdRoad-AI

### AI-Powered Pothole Detection System for the Kurdistan Region of Iraq

KurdRoad-AI is an artificial intelligence system designed to detect and identify potholes on roads using computer vision and YOLO object detection.

The system can analyze both **road images and videos**, locate potholes using bounding boxes, and provide detection confidence and severity information.

---

## 🚀 Features

- 📷 **Image Pothole Detection**
- 🎥 **Video Pothole Detection**
- 🕳️ Automatic pothole localization
- 🎯 Detection confidence score
- ⚠️ Pothole severity classification
- ⚡ GPU-accelerated inference
- 🖥️ Interactive Streamlit web application
- 🌍 Focused on roads in Kurdistan and Iraq

---

## 🧠 AI Model

KurdRoad-AI uses the **YOLO object detection framework** with a custom-trained pothole detection model.

The model was trained using road images collected from the Kurdistan Region and Iraq.

### Detection Pipeline

```text
Road Image / Video
        ↓
   YOLO Detection
        ↓
Pothole Localization
        ↓
Confidence Analysis
        ↓
Severity Classification
        ↓
Detection Result
