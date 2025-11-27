import cv2
from src.detection.face_detector import FaceDetector

detector = FaceDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    boxes, scores = detector.detect(frame)
    frame = detector.draw(frame, boxes, scores)
    
    cv2.imshow("Face Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
