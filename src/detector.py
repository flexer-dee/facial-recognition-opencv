import cv2

class FaceDetector:
    # Set to match the 2026may filename
    def __init__(self, model_path="data/face_detection_yunet_2026may.onnx", conf_threshold=0.6):
        # Initialize the YuNet model
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320), # Will dynamically update per frame
            score_threshold=conf_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

    def get_faces(self, frame):
        """Extracts face arrays (bounding box + landmarks + confidence)."""
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        
        # The detect function returns (status, faces)
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []