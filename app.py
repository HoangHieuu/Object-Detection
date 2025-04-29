import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

st.sidebar.title("Settings Panel")

#Model selection
model_choice = st.sidebar.selectbox(
    "Choose YOLOv8 Model",
    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0
)

# Load selected model
model = YOLO(model_choice)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Top-N detections input
top_n = st.sidebar.number_input("Select Top-N Detections to Show", min_value=1, value=5, step=1)

# Resize option
resize_width = st.sidebar.selectbox(
    "Resize Width (px)",
    options=[640, 416, 320],
    index=0
)

resize_height = resize_width  # Keep it square for simplicity

# Main page content
st.title("🔍 Object Detection with YOLOv8 + Streamlit (Sidebar Version)")
st.write("Upload one or multiple images, adjust settings on the sidebar, and detect objects!")

# Upload multiple images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    progress = st.progress(0)  # Initialize progress bar

    for idx, uploaded_file in enumerate(uploaded_files):
        st.header(f"{uploaded_file.name}")

        # Read and auto-resize image
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((resize_width, resize_height))  # Resize based on user setting
        img_array = np.array(image)

        # Show the original resized image
        st.image(image, caption=f'Uploaded & Resized ({resize_width}x{resize_height})', use_container_width=True)

        # Run YOLO prediction
        st.write(f"Detecting objects with confidence > {confidence_threshold}")
        results = model.predict(img_array, conf=confidence_threshold)

        # Draw bounding boxes
        result_img = results[0].plot()

        # Display result image
        st.image(result_img, caption='Detected Objects', use_container_width=True)

        # Extract detections
        names = results[0].names
        detections = results[0].boxes.data.cpu().numpy()

        if len(detections) > 0:
            # Prepare dataframe
            detected_objects = []
            for det in detections:
                confidence = float(det[4])
                class_id = int(det[5])
                label = names[class_id]
                detected_objects.append({"Class": label, "Confidence": confidence})

            df = pd.DataFrame(detected_objects)

            # Sort by confidence and keep Top-N
            df = df.sort_values(by="Confidence", ascending=False).head(top_n)

            st.subheader("Top-N Detected Objects")
            st.dataframe(df)

        else:
            st.info("No objects detected.")

        # Allow downloading the result image
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        result_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Result Image",
            data=byte_im,
            file_name=f"detected_{uploaded_file.name}",
            mime="image/jpeg"
        )

        # Update the progress bar
        progress.progress((idx + 1) / len(uploaded_files))

    st.success("✅ Detection Completed!")
