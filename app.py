import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO(r"C:\Users\pawan\Facial\model\best (2).pt")  


st.title("Real-Time Face Detection with YOLO")
st.text("Using the trained YOLO model to detect specific faces in real-time")


cap = cv2.VideoCapture(0)


frame_window = st.image([])


while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        st.warning("Camera not found or unable to access.")
        break

    
    frame = cv2.resize(frame, (640, 480))

    
    results = model(frame)

    
    for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        
       
        class_id = int(cls)
        label = f"{model.names[class_id]}: {conf:.2f}" 
        
       
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)  


cap.release()
cv2.destroyAllWindows()
