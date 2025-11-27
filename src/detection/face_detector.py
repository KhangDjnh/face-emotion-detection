import cv2
import mediapipe as mp
from src.config import FACE_DET_CONFIDENCE


class FaceDetector:
    """
    Face Detector sử dụng MediaPipe.
    Trả về bounding boxes theo chuẩn (x1, y1, x2, y2) theo pixel.
    """

    def __init__(self, min_detection_confidence=FACE_DET_CONFIDENCE):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):
        """
        Detect faces trên frame (BGR).
        Return:
            - list bounding boxes [(x1, y1, x2, y2), ...]
            - list scores [float, ...]
        """
        if frame is None:
            return [], []

        height, width = frame.shape[:2]

        # MediaPipe đọc RGB → convert trước
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        boxes = []
        scores = []

        if not result.detections:
            return boxes, scores

        for det in result.detections:
            bbox = det.location_data.relative_bounding_box

            x1 = int(bbox.xmin * width)
            y1 = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            x2 = x1 + w
            y2 = y1 + h

            boxes.append((x1, y1, x2, y2))
            scores.append(det.score[0])

        return boxes, scores

    def draw(self, frame, boxes, scores):
        """
        Vẽ bounding box lên frame (debug, visualization).
        """
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        return frame
