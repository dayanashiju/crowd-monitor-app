import streamlit as st
import cv2
from ultralytics import YOLO
import time

st.set_page_config(layout="wide")
st.title("📊 IoT Crowd Density Monitoring System")

# Load YOLO
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Camera
cap = cv2.VideoCapture(0)

col1, col2 = st.columns([2,1])

video_placeholder = col1.empty()
count_box = col2.empty()
level_box = col2.empty()

run = st.checkbox("Start Monitoring")

if run:

    for _ in range(1000):

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.resize(frame, (640,480))

        results = model(frame, imgsz=640, conf=0.2)

        people_count = 0

        for r in results:
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    if int(r.boxes.cls[i]) == 0:
                        people_count += 1
                        x1,y1,x2,y2 = map(int, r.boxes.xyxy[i])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Crowd Level
        if people_count <= 10:
            level = "LOW"
            level_color = (0,255,0)
        elif people_count <= 25:
            level = "MEDIUM"
            level_color = (0,165,255)
        else:
            level = "HIGH"
            level_color = (0,0,255)

        # 🔥 Add background rectangle for visibility
        cv2.rectangle(frame, (0,0), (350,110), (0,0,0), -1)

        # 🔥 Show People Count on video
        cv2.putText(frame,
                    f"People Count: {people_count}",
                    (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255,255,255),
                    3)

        # 🔥 Show Crowd Level on video
        cv2.putText(frame,
                    f"Crowd Level: {level}",
                    (10,85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    level_color,
                    3)

        # Display Video
        video_placeholder.image(frame, channels="BGR")

        # Dashboard Display
        count_box.metric("👥 People Count", people_count)
        level_box.metric("🚦 Crowd Level", level)

        time.sleep(0.1)


cap.release()
