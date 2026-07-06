# OpenCV Modular Facial Recognition System

A real-time facial recognition and automated attendance tracking system utilizing OpenCV's Deep Neural Network (DNN) module. This project operates entirely on edge-compatible ONNX models, ensuring high computational efficiency without the overhead of extensive external machine learning frameworks.

## Technical Architecture

The system is strictly modular to separate distinct processing stages:

* **Face Detection (YuNet):** Encapsulated within src/detector.py. Processes input frames to output bounding boxes and 5-point facial landmarks using the face_detection_yunet_2026may.onnx weights.
* **Facial Alignment & Feature Extraction (SFace):** Encapsulated within src/encoder.py. Utilizes the 5-point landmarks to align faces and extract 128-dimensional mathematical feature vectors via the face_recognition_sface_2021dec.onnx model.
* **Verification Pipeline:** Identity is verified through Cosine Similarity matching against pre-computed database embeddings, utilizing a standard confidence threshold of 0.363.

## Installation & Setup

1. **Initialize the Environment:**
   Clone the repository and configure the virtual environment.
   > python -m venv .venv
   > .\.venv\Scripts\Activate.ps1
   > pip install opencv-python numpy

2. **Model Asset Acquisition:**
   Create a data/ directory in the project root. Download the raw binary models directly into this folder:
   - YuNet Detector: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx
   - SFace Recognizer: https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognizer_fast.onnx

## Operating Instructions

1. **Populate Database:** Place clear reference photographs into the ImagesAttendance/ directory. Name each file according to the subject's identity (e.g., Paul_Kagame.jpg).
2. **Execute Pipeline:** Start the application orchestrator:
   > python AttendanceProject.py
3. **Graceful Shutdown:** Ensure the video stream window is in focus and press "q" to release hardware resources and exit.