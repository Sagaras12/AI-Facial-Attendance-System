import streamlit as st
import face_recognition
import cv2
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Page title
st.set_page_config(
    page_title="AI Facial Attendance System",
    page_icon="🎯",
    layout="centered"
)
# Sidebar
st.sidebar.title("📌 Navigation")

st.sidebar.info(
    """
    AI Facial Attendance System
    
    Built using:
    - Python
    - OpenCV
    - Streamlit
    - Face Recognition
    """
)

st.title("🎯 AI Facial Attendance System")
st.write("Real-time face recognition attendance system using AI and OpenCV")

# Load encodings
if not os.path.exists("encodings.pkl"):
    st.error("encodings.pkl not found! Please run save_encodings.py first.")
    st.stop()

with open("encodings.pkl", "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_names = data["names"]

# Attendance file
attendance_file = "attendance.csv"

# Create attendance file if not exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# Start camera button
start = st.button("Start Camera")
stop = st.button("Stop Camera")

FRAME_WINDOW = st.image([])

# Prevent repeated attendance
marked_names = set()

if start:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam")
        st.stop()

    while True:

        ret, frame = cap.read()

        if not ret:
            st.error("Failed to access webcam")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)

        # Encode faces
        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations
        )

        for (top, right, bottom, left), face_encoding in zip(
            face_locations,
            face_encodings
        ):

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding
            )

            name = "Unknown"
            display_name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            if len(face_distances) > 0:

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:

                    name = known_names[best_match_index]

                    confidence = round(
                        (1 - face_distances[best_match_index]) * 100,
                        2 
                    )

                    display_name = f"{name} ({confidence}%)"
                else:
                    display_name = "Unknown"

            # Rectangle color
            color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                color,
                2
            )

            # Display name
            cv2.putText(
                frame,
                display_name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            # Attendance marking
            if name != "Unknown" and name not in marked_names:

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                existing_data = pd.read_csv(attendance_file)

                already_marked = existing_data[
                    (existing_data["Name"] == name) &
                    (existing_data["Date"] == date)
                ]

                if already_marked.empty:

                    new_entry = pd.DataFrame(
                        [[name, date, time]],
                        columns=["Name", "Date", "Time"]
                    )

                    new_entry.to_csv(
                        attendance_file,
                        mode='a',
                        header=False,
                        index=False
                    )

                    st.toast(f"✅ Attendance marked for {name}")

                else:
                    st.info(f"{name} already marked today")

                marked_names.add(name)

        # Show webcam frame
        FRAME_WINDOW.image(frame, channels="BGR")
        if stop:
            cap.release()
            st.warning("Camera stopped")
            break

    cap.release()

# Show attendance table
st.markdown("---")
st.subheader("📄 Attendance Dashboard")
if os.path.exists(attendance_file):

    attendance_data = pd.read_csv(attendance_file)

    total_records = len(attendance_data)

    unique_people = attendance_data["Name"].nunique()

    col1, col2 = st.columns(2)

    col1.metric("Total Attendance", total_records)

    col2.metric("Unique People", unique_people)

    csv = attendance_data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Attendance CSV",
        data=csv,
        file_name="attendance.csv",
        mime="text/csv"
)


st.markdown("---")

st.caption(
    "🚀 Developed by Sagar Gupta | AI Facial Attendance System"
)