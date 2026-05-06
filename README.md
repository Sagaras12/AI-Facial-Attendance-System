# AI-Based Facial Attendance System 🎯

A real-time AI-powered facial attendance system built using Python, OpenCV, and Face Recognition.

This project detects and recognizes faces using a webcam and automatically marks attendance with date and time.

---

# 🚀 Features

✅ Real-time face detection using webcam  
✅ Face recognition using AI  
✅ Automatic attendance marking  
✅ Saves attendance in CSV file  
✅ Prevents duplicate attendance  
✅ Handles unknown faces  
✅ Fast startup using saved encodings (`encodings.pkl`)  
✅ Optimized using Pickle serialization  

---

# 🛠️ Technologies Used

- Python
- OpenCV
- face_recognition
- NumPy
- Pandas
- Pickle

---

# 📂 Project Structure

```plaintext
face_attendance_system/
│
├── dataset/
│
├── attendance.csv
├── encodings.pkl
│
├── save_encodings.py
├── attendance_system.py
│
├── requirements.txt
└── README.md
```

---

# ⚡ Installation Process

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-Facial-Attendance-System.git
```

---

## 2️⃣ Open Project Folder

```bash
cd AI-Facial-Attendance-System
```

---

## 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# 📸 Add Face Images

Inside the `dataset` folder create a folder with your name.

Example:

```plaintext
dataset/
└── Sagar/
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
```

Add multiple face images with different angles.

---

# 🧠 Generate Face Encodings

Run:

```bash
python save_encodings.py
```

This creates:

```plaintext
encodings.pkl
```

which stores facial data for fast recognition.

---

# ▶️ Run Attendance System

Run:

```bash
python attendance_system.py
```

---

# 📄 Attendance Output

Attendance is automatically stored in:

```plaintext
attendance.csv
```

Example:

```plaintext
Name,Date,Time
Sagar,2026-05-04,22:34:31
```

---

# 🔥 Future Improvements

- Streamlit UI
- SQL Database Integration
- Multiple User Management
- Confidence Percentage Display
- Email Notifications
- Attendance Dashboard

---

# 👨‍💻 Author

Sagar Gupta  
B.Sc Artificial Intelligence Student