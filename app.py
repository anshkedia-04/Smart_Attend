# app.py
import streamlit as st
st.set_page_config(layout="centered")  # ✅ Must be first Streamlit command

import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Constants
HAAR_FILE = 'haarcascade_frontalface_default.xml'
DATASET_DIR = 'dataset'
TRAINER_FILE = 'trainer.yml'
STUDENTS_FILE = 'students.csv'
ATTENDANCE_DIR = 'attendance'
PASSWORD = "teacher123"

# Create required directories
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Slots for attendance
NUM_SLOTS = 6
SLOTS = [f"Slot {i+1}" for i in range(NUM_SLOTS)]

# Ensure students.csv exists
if not os.path.exists(STUDENTS_FILE):
    pd.DataFrame(columns=["Roll", "Name"]).to_csv(STUDENTS_FILE, index=False)

# Get today's attendance file path
def get_attendance_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

# Load attendance or create fresh file
def load_attendance():
    file = get_attendance_file()
    if not os.path.exists(file):
        df = pd.read_csv(STUDENTS_FILE)
        for slot in SLOTS:
            df[slot] = "Absent"
        df.to_csv(file, index=False)
    return pd.read_csv(file)

# Save updated attendance
def save_attendance(df):
    df.to_csv(get_attendance_file(), index=False)

# Recognize face and mark attendance
def recognize_and_mark_attendance(slot):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    face_cascade = cv2.CascadeClassifier(HAAR_FILE)

    students_df = pd.read_csv(STUDENTS_FILE)
    attendance_df = load_attendance()
    marked = []

    cap = cv2.VideoCapture(0)
    st.warning("📸 Camera started. Press 'Q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi)

            if conf < 60:
                roll_str = str(id_)
                student_row = students_df[students_df['Roll'] == roll_str]

                if not student_row.empty:
                    name = student_row['Name'].values[0]
                else:
                    # Automatically add unknown ID to students.csv
                    name = f"Student_{roll_str}"
                    new_student = pd.DataFrame([[roll_str, name]], columns=["Roll", "Name"])
                    students_df = pd.concat([students_df, new_student], ignore_index=True)
                    students_df.to_csv(STUDENTS_FILE, index=False)
                    st.info(f"➕ Added ID {roll_str} as {name} to student records.")

                    # Also add to today's attendance file
                    if roll_str not in attendance_df['Roll'].values:
                        new_row = pd.Series([roll_str, name] + ["Absent"] * len(SLOTS), index=attendance_df.columns)
                        attendance_df = pd.concat([attendance_df, new_row.to_frame().T], ignore_index=True)

                # Now mark attendance
                if roll_str in attendance_df['Roll'].values:
                    current_status = attendance_df.loc[attendance_df['Roll'] == roll_str, slot].values[0]
                    if current_status == "Absent":
                        attendance_df.loc[attendance_df['Roll'] == roll_str, slot] = "Present"
                        st.success(f"✅ Marked Present: {roll_str} - {name}")
                        marked.append((roll_str, name))


                cv2.putText(frame, f"ID: {roll_str}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Mark Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_attendance(attendance_df)
    if not marked:
        st.error("❌ No known face detected.")

# Teacher dashboard
def view_attendance():
    st.subheader("📋 Attendance Dashboard")
    df = load_attendance()
    st.dataframe(df)

# Streamlit UI
st.title("🎓 AI-Based Attendance System")

tab = st.selectbox("Choose Mode", ["📌 Mark Attendance", "🔐 Teacher Panel"])

if tab == "📌 Mark Attendance":
    slot = st.selectbox("Select Slot", SLOTS)
    if st.button("📷 Start Camera and Mark Attendance"):
        recognize_and_mark_attendance(slot)

elif tab == "🔐 Teacher Panel":
    pwd = st.text_input("Enter Teacher Password", type="password")
    if pwd == PASSWORD:
        view_attendance()
    elif pwd:
        st.error("Incorrect password.")
