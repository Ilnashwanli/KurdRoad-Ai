import streamlit as st
from PIL import Image
from detector import detect_potholes
from ultralytics import YOLO
import tempfile
import cv2
import os

st.set_page_config(
    page_title="KurdRoad-AI",
    page_icon="🛣️",
    layout="centered"
)

st.title("🛣️ KurdRoad-AI")
st.subheader("AI-Powered Pothole Detection")

st.write(
    "Upload a road image or video and let AI identify potholes."
)

# Load video model
VIDEO_MODEL_PATH = r"P:\Ai\KurdRoad-Ai\runs\pothole_v4\weights\best.pt"


# =========================================================
# IMAGE DETECTION
# =========================================================

st.header("📷 Image Detection")

uploaded_image = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
    key="image_upload"
)

if uploaded_image is not None:

    image = Image.open(uploaded_image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Potholes", key="image_button"):

        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        with st.spinner("AI is analyzing the road..."):

            result, pothole_count, confidence, severity = (
                detect_potholes("temp_image.jpg")
            )

        result_image = result.plot()

        st.subheader("🤖 Detection Result")

        st.image(
            result_image,
            channels="BGR",
            use_container_width=True
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🕳️ Potholes",
                pothole_count
            )

        with col2:
            st.metric(
                "🎯 Confidence",
                f"{confidence * 100:.1f}%"
            )

        with col3:
            st.metric(
                "⚠️ Severity",
                severity
            )

        if pothole_count == 0:
            st.success("No potholes detected.")

        elif severity == "High":
            st.error("High pothole density detected.")

        elif severity == "Medium":
            st.warning("Medium pothole density detected.")

        else:
            st.info("Low pothole density detected.")


# =========================================================
# VIDEO DETECTION
# =========================================================

st.divider()

st.header("🎥 Video Detection")

uploaded_video = st.file_uploader(
    "Upload a road video",
    type=["mp4", "webm", "avi", "mov", "mkv"],
    key="video_upload"
)

if uploaded_video is not None:

    if st.button("🎬 Detect Potholes in Video", key="video_button"):

        with st.spinner("AI is analyzing the video..."):

            # Save uploaded video temporarily
            video_extension = os.path.splitext(
                uploaded_video.name
            )[1]

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=video_extension
            )

            input_file.write(uploaded_video.getbuffer())
            input_file.close()

            input_path = input_file.name

            # Open video
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("❌ Could not open the video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Output video
            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_file.name
            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # Load model
            model = YOLO(VIDEO_MODEL_PATH)

            total_potholes = 0
            max_confidence = 0

            # Process video
            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(
                    source=frame,
                    conf=0.15,
                    device=0,
                    verbose=False
                )

                result = results[0]

                # Count potholes
                if result.boxes is not None:
                    count = len(result.boxes)

                    total_potholes += count

                    if count > 0:
                        frame_confidence = float(
                            result.boxes.conf.max()
                        )

                        max_confidence = max(
                            max_confidence,
                            frame_confidence
                        )

                # Draw detections
                annotated_frame = result.plot()

                out.write(annotated_frame)

            cap.release()
            out.release()

            # Clean input file
            try:
                os.remove(input_path)
            except:
                pass

        st.success("✅ Video detection completed!")

        st.subheader("🤖 Detection Result")

        st.video(output_path)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🕳️ Detected Potholes",
                total_potholes
            )

        with col2:
            st.metric(
                "🎯 Best Confidence",
                f"{max_confidence * 100:.1f}%"
            )

        if total_potholes == 0:
            st.success("No potholes detected in the video.")

        elif max_confidence >= 0.70:
            st.error("⚠️ Strong pothole detection.")

        elif max_confidence >= 0.40:
            st.warning("⚠️ Moderate pothole detection.")

        else:
            st.info("ℹ️ Low-confidence pothole detection.")