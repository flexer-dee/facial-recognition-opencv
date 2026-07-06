# OpenCV Modular Facial Recognition System

A lightweight, real-time facial recognition and attendance tracking system built entirely with OpenCV DNN.

## Project Structure
FacialRecognitionPython/
├── AttendanceProject.py       # Main app
├── data/                      # Model files
│   ├── face_detection_yunet_2026may.onnx
│   └── face_recognition_sface_2021dec.onnx
├── ImagesAttendance/          # Drop reference photos here
├── src/                       # Modules
│   ├── detector.py
│   └── encoder.py
└── Attendance.csv             # Logs

## Raw Model Download Links
* YuNet Detector: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx
* SFace Recognizer: https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognizer_fast.onnx

## Quick Setup (Windows)
1. python -m venv .venv
2. .\.venv\Scripts\Activate.ps1
3. pip install opencv-python numpy
4. Add images to 'ImagesAttendance/'
5. python AttendanceProject.py

Press 'q' in the camera window to safely exit.