# 🎯 SMART-ATTEND  
> **AI-powered Face Recognition Attendance System**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Backend-ff4b4b?style=for-the-badge&logo=streamlit)
![React](https://img.shields.io/badge/React-Frontend-blue?style=for-the-badge&logo=react)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%2C%20MobileNet%2C%20FaceNet-orange?style=for-the-badge&logo=tensorflow)

---

## 📌 Overview
**SMART-ATTEND** is a **real-time face recognition attendance system** built with cutting-edge AI models.  
It offers a **fast, accurate, and contactless** way to record attendance using a webcam feed, making it ideal for classrooms, offices, and events.

We experimented with **three models**:
- 🧠 **Custom CNN**
- ⚡ **MobileNet**
- 🚀 **FaceNet (Winner!)**

FaceNet achieved the highest accuracy (**96.5%**) on our custom dataset.

---

## ✨ Features
✅ Real-time face detection and recognition  
✅ Attendance logging with timestamps  
✅ High accuracy with FaceNet embeddings  
✅ Easy-to-use web interface (React.js)  
✅ REST API for backend processing (Flask)  
✅ Modular design for future expansion  

---

## 📊 Model Performance
| Model      | Accuracy (%) | Notes |
|------------|--------------|-------|
| CNN        | 68.37%       | Basic custom model |
| MobileNet  | 82.45%       | Lightweight & fast |
| **FaceNet** | **96.50%**  | Best performance, robust to variations |

---

## 🛠 Tech Stack
**Frontend:** React.js, react-webcam  
**Backend:** Flask, Python  
**AI Models:** CNN, MobileNet, FaceNet (InceptionResnetV1)  
**Libraries:** OpenCV, MediaPipe, NumPy, scikit-learn, facenet-pytorch  

