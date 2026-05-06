import face_recognition
import os
import numpy as np

dataset_path = "dataset"

known_encodings = []
known_names = []

# Loop through dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        # Load image
        image = face_recognition.load_image_file(image_path)

        # Get face encodings
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"No face found in {image_path}")

print("Encoding Complete!")
print(f"Total faces encoded: {len(known_encodings)}")