import cv2

def draw_bbox(frame, bbox, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def draw_landmarks(frame, landmarks, color=(0,0,255)):
    for x, y, _ in landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, color, -1)

def draw_text(frame, text, position=(10,30), color=(0,255,0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
