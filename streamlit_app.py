import streamlit as st
import face_recognition
import cv2
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sqlite3

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Facial Attendance System",
    page_icon="🎯",
    layout="centered"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Navigation")

st.sidebar.info(
    """
    AI Facial Attendance System

    Built using:
    - Python
    - OpenCV
    - Streamlit
    - Face Recognition
    - SQLite
    """
)

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("🎯 AI Facial Attendance System")

st.write(
    "Real-time face recognition attendance system using AI and OpenCV"
)

# -----------------------------
# LOAD ENCODINGS
# -----------------------------
if not os.path.exists("encodings.pkl"):
    st.warning("Demo Version: Face recognition disabled because no dataset is included.")
    st.stop()

with open("encodings.pkl", "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_names = data["names"]

# -----------------------------
# SQLITE DATABASE
# -----------------------------
conn = sqlite3.connect(
    "attendance.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        time TEXT
    )
    """
)

conn.commit()

# -----------------------------
# BUTTONS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    start = st.button("▶️ Start Camera")

with col2:
    stop = st.button("⏹️ Stop Camera")

FRAME_WINDOW = st.image([])

# Prevent repeated messages
marked_names = set()

# -----------------------------
# CAMERA START
# -----------------------------
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

        # Encode detected faces
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
                    name = name.capitalize()

                    confidence = round(
                        (1 - face_distances[best_match_index]) * 100,
                        2
                    )

                    display_name = f"{name} ({confidence}%)"

            # Rectangle color
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                color,
                2
            )

            # Display label
            cv2.putText(
                frame,
                display_name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            # -----------------------------
            # ATTENDANCE LOGIC
            # -----------------------------
            if name != "Unknown" and name not in marked_names:

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                # Check duplicate attendance
                cursor.execute(
                    """
                    SELECT * FROM attendance
                    WHERE name = ? AND date = ?
                    """,
                    (name, date)
                )

                already_marked = cursor.fetchall()

                if len(already_marked) == 0:

                    cursor.execute(
                        """
                        INSERT INTO attendance (name, date, time)
                        VALUES (?, ?, ?)
                        """,
                        (name, date, time)
                    )

                    conn.commit()

                    st.toast(f"✅ Attendance marked for {name}")

                else:
                    st.info(f"{name} already marked today")

                marked_names.add(name)

        # Show webcam
        FRAME_WINDOW.image(frame, channels="BGR")

        # Stop camera
        if stop:
            cap.release()
            st.warning("Camera stopped")
            break

    cap.release()

# -----------------------------
# DASHBOARD
# -----------------------------
st.markdown("---")

st.subheader("📄 Attendance Dashboard")

attendance_data = pd.read_sql_query(
    "SELECT * FROM attendance",
    conn
)

if len(attendance_data) > 0:
    
    st.dataframe(attendance_data)
else:
    st.info("No attendance records found")

# Metrics
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Attendance", len(attendance_data))

with col2:
    if len(attendance_data) > 0:
        st.metric(
            "Unique People",
            attendance_data["name"].nunique()
        )
    else:
        st.metric("Unique People", 0)

# Show table


# -----------------------------
# DOWNLOAD BUTTON
# -----------------------------
csv = attendance_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Export Attendance Report",
    data=csv,
    file_name="attendance.csv",
    mime="text/csv"
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.caption(
    "🚀 Developed by Sagar Gupta | AI Facial Attendance System"
)

