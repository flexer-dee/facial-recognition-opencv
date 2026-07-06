# Real-Time Modular Facial Recognition & Attendance System

A lightweight, real-time facial recognition and automated attendance tracking system built entirely on top of OpenCV's Deep Neural Network (DNN) module.

## Technical Architecture
* **Face Detection (YuNet):** Handled via \src/detector.py\ using the ultra-light edge model (\ace_detection_yunet_2026may.onnx\).
* **Facial Alignment & Feature Extraction (SFace):** Handled via \src/encoder.py\ using the \ace_recognition_sface_2021dec.onnx\ model to extract 128-dimensional mathematical feature vectors.
* **Verification Pipeline:** Identity is verified using Cosine Similarity matching with a default standard boundary threshold of 0.363.

## Installation & Setup
1. Clone the repository and initialize your environment:
   \\\powershell
   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   pip install opencv-python numpy
   \\\" Add-Content -Path 
2. Ensure the raw binary models are placed in the \data/\ directory.
