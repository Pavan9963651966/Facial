import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load your custom-trained YOLO model
model = YOLO(r"C:\Users\pawan\Facial\model\best (2).pt")  # Replace with the path to your 'best.pt' model

# Streamlit app title
st.title("Real-Time Face Detection with YOLO")
st.text("Using the trained YOLO model to detect specific faces in real-time")

# Start video capture from the laptop's camera
cap = cv2.VideoCapture(0)

# Streamlit video display
frame_window = st.image([])

# Main loop to read frames from the webcam and perform detection
while cap.isOpened():
    ret, frame = cap.read()  # Capture each frame
    if not ret:
        st.warning("Camera not found or unable to access.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO model on the frame to detect faces
    results = model(frame)

    # Draw bounding boxes and add label for each detected face
    for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        
        # Get class name from the model's class names (ensure your model has them set up correctly)
        class_id = int(cls)
        label = f"{model.names[class_id]}: {conf:.2f}"  # Display class name and confidence
        
        # Draw the bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the frame color for Streamlit display (BGR to RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)  # Display the frame in the app

# Release the video capture when done
cap.release()
cv2.destroyAllWindows()
