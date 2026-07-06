import cv2

class FaceEncoder:
    # Set to match the 2021dec filename
    def __init__(self, model_path="data/face_recognition_sface_2021dec.onnx"):
        # Initialize the SFace recognition model
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=model_path,
            config=""
        )

    def get_embedding(self, frame, face_data):
        """Aligns the face and extracts the mathematical feature vector."""
        # Align the face using the 5-point landmarks provided by the detector
        aligned_face = self.recognizer.alignCrop(frame, face_data)
        
        # Extract the continuous mathematical features (embeddings)
        feature = self.recognizer.feature(aligned_face)
        return feature

    def compare(self, feature1, feature2, threshold=0.363):
        """Compares two embeddings using Cosine Similarity. 
           Scores higher than 0.363 are considered a match."""
        score = self.recognizer.match(
            feature1, 
            feature2, 
            cv2.FaceRecognizerSF_FR_COSINE
        )
        return score > threshold