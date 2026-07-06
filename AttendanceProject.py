import cv2
import numpy as np
import os
from datetime import datetime

# Import our custom OpenCV modules with a fallback 
# just in case the files aren't in a 'src' folder
try:
    from src.detector import FaceDetector
    from src.encoder import FaceEncoder
except ModuleNotFoundError:
    from detector import FaceDetector
    from encoder import FaceEncoder

def findEncodings(images, detector, encoder):
    encodeList = []
    for img in images:
        # 1. Detect the face in the image
        faces = detector.get_faces(img)
        
        # 2. Ensure a face was found before trying to encode
        if len(faces) > 0:
            # We assume the most prominent face is the first one in the array
            encode = encoder.get_embedding(img, faces[0])
            encodeList.append(encode)
        else:
            print("Warning: No face detected in one of the source images.")
            encodeList.append(None)
            
    return encodeList

def markAttendance(name):
    # 'a+' safely creates the file if it doesn't exist and appends to it
    with open('Attendance.csv', 'a+') as f:
        f.seek(0) # Move to the beginning of the file to read existing entries
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def main():
    path = 'ImagesAttendance'
    images = []
    classNames = []
    
    # Ensure the directory exists to prevent crashes
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created '{path}' directory. Please add images and restart.")
        return

    myList = os.listdir(path)
    print(f"Found images: {myList}")

    # Read images and extract names
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
            
    print(f"Classes: {classNames}")

    # Initialize the OpenCV models (matching your local downloads)
    detector = FaceDetector(model_path="data/face_detection_yunet_2026may.onnx")
    encoder = FaceEncoder(model_path="data/face_recognition_sface_2021dec.onnx")

    encodeListKnown = findEncodings(images, detector, encoder)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Detect all faces in the current frame
        facesCurFrame = detector.get_faces(img)
        
        if len(facesCurFrame) > 0:
            for faceLoc in facesCurFrame:
                # Extract the mathematical embedding for the found face
                encodeFace = encoder.get_embedding(img, faceLoc)
                
                matchScores = []
                # Compare against all our known student/employee encodings
                for knownEncode in encodeListKnown:
                    if knownEncode is not None:
                        # Cosine similarity: closer to 1.0 is a better match
                        score = encoder.recognizer.match(knownEncode, encodeFace, cv2.FaceRecognizerSF_FR_COSINE)
                        matchScores.append(score)
                    else:
                        matchScores.append(0.0)
                
                # Find the highest similarity score
                matchIndex = np.argmax(matchScores)
                bestScore = matchScores[matchIndex]
                
                # 0.363 is the standard threshold for SFace Cosine matching
                if bestScore >= 0.363:
                    name = classNames[matchIndex].upper()
                else:
                    name = "UNKNOWN"
                
                # The FaceDetector returns [x, y, w, h] as the first 4 items
                box = list(map(int, faceLoc[:4]))
                x1, y1, w, h = box[0], box[1], box[2], box[3]
                x2, y2 = x1 + w, y1 + h
                
                # Draw the bounding box and name tag
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                if name != "UNKNOWN":
                    markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()