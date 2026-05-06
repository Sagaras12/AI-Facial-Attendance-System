import face_recognition
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

# Dataset path
dataset_path = "dataset"

# Prevent repeated terminal messages
printed_once = set()


# Load saved encodings
with open("encodings.pkl", "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_names = data["names"]

print("Encodings loaded instantly!")


# Attendance file
attendance_file = "attendance.csv"

# Create CSV if not exists
if not os.path.exists(attendance_file):

    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# Load existing attendance
existing_data = pd.read_csv(attendance_file)

# Start webcam
cap = cv2.VideoCapture(0)

# Webcam check
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
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

        face_distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )

        if len(face_distances) > 0:

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]

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

        # Put name text
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        # Attendance logic
        if name != "Unknown":

            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # Check duplicate attendance
            already_marked = existing_data[
                (existing_data["Name"] == name) &
                (existing_data["Date"] == date)
            ]

            # Mark attendance
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

                # Update dataframe
                existing_data = pd.concat(
                    [existing_data, new_entry],
                    ignore_index=True
                )

                print(f"Attendance marked for {name}")

            else:

                # Print only once per run
                if name not in printed_once:

                    print(f"{name} already marked today")

                    printed_once.add(name)

    # Show webcam
    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(1)

    # ESC key
    if key == 27:
        break

# Release resources
cap.release()

cv2.destroyAllWindows()