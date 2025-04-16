import cv2
import numpy as np
import torch
import os
import time
import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/model.pth'
if not os.path.exists(model_path):
    st.error("Model file not found")
    st.stop()
try:
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if "run" not in st.session_state:
    st.session_state.run = False

def start_detection():
    st.session_state.run = True

def stop_detection():
    st.session_state.run = False

st.title("Real-time Facial Emotion Recognition")
col1, col2 = st.columns(2)
with col1:
    st.button("Start Emotion Detection", on_click=start_detection)
with col2:
    st.button("Stop Emotion Detection", on_click=stop_detection)

frame_placeholder = st.empty()

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        st.stop()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        mirrored_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_tensor = torch.from_numpy(face_resized).float() / 255.0
            face_tensor = (face_tensor - 0.5) / 0.5
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)
            face_tensor = face_tensor.to(device)
            try:
                with torch.no_grad():
                    prediction = model(face_tensor)
                    emotion_index = torch.argmax(prediction, dim=1).item()
                    emotion = emotion_labels[emotion_index]
                cv2.rectangle(mirrored_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(mirrored_frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        frame_placeholder.image(cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        time.sleep(0.03)
    cap.release()
