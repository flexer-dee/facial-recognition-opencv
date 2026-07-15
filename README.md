# Real-Time Modular Facial Recognition & Attendance System

A lightweight, real-time facial recognition and automated attendance tracking system built entirely on top of **OpenCV's Deep Neural Network (DNN) module**. 

By utilizing optimized ONNX models directly through OpenCV, this system achieves exceptional frame rates on standard CPU hardware without requiring heavy machine learning frameworks like TensorFlow, PyTorch, or dlib.

---

## Technical Architecture

The system is designed with a strictly modular architecture to enforce clean separation of concerns:

```text
FacialRecognitionPython/
├── AttendanceProject.py       # Application orchestration & webcam pipeline
├── data/                      # Local storage for neural network weights
│   ├── face_detection_yunet_2026may.onnx
│   └── face_recognition_sface_2021dec.onnx
├── ImagesAttendance/          # Input directory for authorized personnel photos
├── src/                       # Core engine modules
│   ├── detector.py            # YuNet face localization encapsulation
│   └── encoder.py             # SFace alignment and embedding extraction
└── Attendance.csv             # Automated tabular data logging

### 1. Face Detection (YuNet)
* **Model:** `face_detection_yunet_2026may.onnx` (~224 KB)
* **Role:** Located in `src/detector.py`, this module handles face localization. It processes input frames, dynamically resizes the input layer, and outputs a bounding box alongside 5 critical facial landmarks (eyes, nose, and mouth corners).
* **Performance:** Optimized for edge computing and mobile processors, offering high-speed tracking even with multiple faces in frame.

### 2. Facial Alignment & Feature Extraction (SFace)
* **Model:** `face_recognition_sface_2021dec.onnx` (~37 MB)
* **Role:** Located in `src/encoder.py`. It takes the 5-point landmarks from YuNet to warp and crop the face into a normalized position. It then passes the aligned face through a convolutional neural network to generate a 128-dimensional mathematical feature vector (embedding).

### 3. Verification Pipeline
* **Metric:** Cosine Similarity
* **Threshold:** 0.363 (Standard verification boundary)
* Identity is computed by taking the dot product of the runtime face embedding and the pre-computed database embeddings. A similarity score greater than or equal to 0.363 confirms a matching identity.

---

## Installation & Setup

### 1. Environment Configuration
Clone the repository, create a isolated virtual environment, and install core dependencies:
```powershell
git clone [https://github.com/YourUsername/facial-recognition-opencv.git](https://github.com/YourUsername/facial-recognition-opencv.git)
cd facial-recognition-opencv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install opencv-python numpy
### 2. Model Asset Acquisition
Due to file storage limits, model weights must be downloaded locally into the `data/` directory. Ensure you download the **raw binary** files:
* [Download YuNet Detector](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx)
* [Download SFace Recognizer](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognizer_fast.onnx)

---

## Operating Instructions

1. **Populate Reference Database:** Place clear, front-facing portrait images into the `ImagesAttendance` folder. The filename determines the registered name tag (e.g., `Paul_Kagame.jpg` will log as `PAUL_KAGAME`).
2. **Execute System:** Run the application orchestrator:
   ```powershell
   python AttendanceProject.py

3. **Runtime Mechanics:** The webcam feed initializes immediately. Green tracking boxes mark identified individuals, while unregistered profiles display as `UNKNOWN`.
4. **Graceful Shutdown:** Focus the active video window and press the **`q`** key to safely flush system resources and disconnect the camera hardware.

---

## Troubleshooting & Best Practices

* **ModuleNotFoundError (cv2):** Ensure your IDE terminal is using the active virtual environment (`.venv`). If deactivated, rerun the activation script before executing the program.
* **Can't read ONNX file:** This signifies a corrupted model download (usually an HTML wrapper page instead of raw binaries). Delete the file from the `data/` folder and use the direct raw links provided above.
* **Low Identification Confidence:** Ensure reference images have flat lighting, neutral expressions, and no obstructions like heavy sunglasses.