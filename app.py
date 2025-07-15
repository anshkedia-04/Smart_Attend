import streamlit as st
st.set_page_config(layout="centered")

import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
HAAR_FILE = 'haarcascade_frontalface_default.xml'
DATASET_DIR = 'dataset'
MODEL_PATH = 'trained_model/cnn_face_model.h5'
LABELS_PATH = 'trained_model/label_classes.npy'
STUDENTS_FILE = 'students.csv'
ATTENDANCE_DIR = 'attendance'
PASSWORD = "teacher123"
IMG_SIZE = (100, 100)

# Create directories if not exist
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Slots
NUM_SLOTS = 6
SLOTS = [f"Slot {i+1}" for i in range(NUM_SLOTS)]

# Ensure students.csv exists
if not os.path.exists(STUDENTS_FILE):
    pd.DataFrame(columns=["Roll", "Name"]).to_csv(STUDENTS_FILE, index=False)

# Attendance file path
def get_attendance_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

# Load or create attendance
def load_attendance():
    file = get_attendance_file()
    if not os.path.exists(file):
        df = pd.read_csv(STUDENTS_FILE)
        for slot in SLOTS:
            df[slot] = "Absent"
        df.to_csv(file, index=False)
    return pd.read_csv(file)

# Save attendance
def save_attendance(df):
    df.to_csv(get_attendance_file(), index=False)

# Recognize and mark attendance
def recognize_and_mark_attendance(slot):
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_FILE)
    df = load_attendance()
    marked = []

    cap = cv2.VideoCapture(0)
    st.warning("Press 'Q' on webcam window to stop recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, IMG_SIZE)
            face_input = img_to_array(face_resized) / 255.0
            face_input = np.expand_dims(face_input, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)  # shape = (1, 100, 100, 1)

            pred = model.predict(face_input)
            confidence = np.max(pred)
            pred_index = np.argmax(pred)
            name_label = labels[pred_index]

            if confidence > 0.85:
                try:
                    roll = name_label.split('_')[0]
                    name = '_'.join(name_label.split('_')[1:])
                    student_row = df[df["Roll"].astype(str) == str(roll)]
                except Exception as e:
                    st.error(f"Parsing error: {e}")
                    continue

                if not student_row.empty:
                    already_present = df.at[student_row.index[0], slot] == 'Present'
                    if not already_present:
                        df.at[student_row.index[0], slot] = 'Present'
                        marked_time = datetime.now().strftime("%H:%M:%S")
                        marked.append((roll, name))
                        st.success(f"✅ Marked: {roll} - {name} at {marked_time} in {slot}")
                    else:
                        marked_time = datetime.now().strftime("%H:%M:%S")
                        st.info(f"⚠️ {name} already marked as Present in {slot}.")

                # Show on camera
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{name}, {slot}, {datetime.now().strftime('%H:%M:%S')}"
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("Mark Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_attendance(df)

    if not marked:
        st.error("❌ No known face detected.")

# View attendance (Teacher panel)
import time

def view_attendance():
    st.subheader("📋 Attendance Dashboard")
    df = load_attendance()
    st.dataframe(df)

    # Auto-refresh every 10 seconds (optional)
    time.sleep(10)
    st.experimental_rerun()

# Streamlit UI
st.title("🎓 SmartAttend - AI-Based Attendance System")

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
