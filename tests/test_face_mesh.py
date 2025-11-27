import cv2
from src.landmarks.face_mesh import FaceMeshExtractor

cap = cv2.VideoCapture(0)
extractor = FaceMeshExtractor()

while True:
    ret, frame = cap.read()
    data = extractor.extract_landmarks(frame)
    if data:
        print("Landmarks shape:", data['landmarks'].shape)
        print("EAR:", data['ear'], "MAR:", data['mar'], "Head Pose:", data['head_pose'])
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
