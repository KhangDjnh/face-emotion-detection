import cv2
import mediapipe as mp
import numpy as np

class FaceMeshExtractor:
    """
    MediaPipe Face Mesh extractor.
    Trả về landmarks (468,3) và các đặc trưng: EAR, MAR, head pose approximation.
    """

    def __init__(self,
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, frame):
        """
        Input: frame (BGR)
        Output: dict {
            'landmarks': np.array(468,3) (x,y,z normalized),
            'ear': float,
            'mar': float,
            'head_pose': np.array(3,) pitch/yaw/roll in degrees
        }
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = []
        face_landmarks = results.multi_face_landmarks[0]
        for lm in face_landmarks.landmark:
            # chuyển normalized coordinates về pixel
            x, y, z = lm.x * w, lm.y * h, lm.z * w
            landmarks.append([x, y, z])

        landmarks = np.array(landmarks)  # shape (468,3)

        # Tính các đặc trưng phụ trợ
        ear = self.compute_ear(landmarks)
        mar = self.compute_mar(landmarks)
        head_pose = self.compute_head_pose(landmarks, w, h)

        return {
            'landmarks': landmarks,
            'ear': ear,
            'mar': mar,
            'head_pose': head_pose
        }

    # ---------------- Helper functions ----------------

    def compute_ear(self, landmarks):
        """
        EAR - Eye Aspect Ratio
        Chỉ số nhắm/mở mắt (dựa vào 6 điểm mắt)
        """
        # điểm mắt trái (MediaPipe indices)
        left = [33, 160, 158, 133, 153, 144]
        right = [362, 385, 387, 263, 373, 380]

        def eye_aspect_ratio(eye_points):
            # p1,p2,...p6
            p1 = landmarks[eye_points[1]][1] - landmarks[eye_points[5]][1]
            p2 = landmarks[eye_points[2]][1] - landmarks[eye_points[4]][1]
            p0 = landmarks[eye_points[0]][0] - landmarks[eye_points[3]][0]
            ear_val = (p1 + p2) / (2.0 * p0)
            return ear_val

        ear_left = eye_aspect_ratio(left)
        ear_right = eye_aspect_ratio(right)
        return (ear_left + ear_right) / 2.0

    def compute_mar(self, landmarks):
        """
        MAR - Mouth Aspect Ratio
        Chỉ số mở miệng (dựa vào 8 điểm môi)
        """
        mouth = [61, 291, 81, 311, 78, 308, 13, 14]  # approximated
        p61 = landmarks[mouth[0]]
        p291 = landmarks[mouth[1]]
        p81 = landmarks[mouth[2]]
        p311 = landmarks[mouth[3]]
        p78 = landmarks[mouth[4]]
        p308 = landmarks[mouth[5]]
        p13 = landmarks[mouth[6]]
        p14 = landmarks[mouth[7]]

        vert = np.linalg.norm(p81 - p311) + np.linalg.norm(p78 - p308)
        horz = np.linalg.norm(p61 - p291)
        mar = vert / horz
        return mar

    def compute_head_pose(self, landmarks, w, h):
        """
        Head pose approx: pitch/yaw/roll
        Sử dụng một số điểm cố định trên mặt (MediaPipe 468 landmarks)
        """
        # chọn 6 điểm cơ bản
        image_points = np.array([
            landmarks[1][:2],    # Nose tip
            landmarks[33][:2],   # left eye corner
            landmarks[263][:2],  # right eye corner
            landmarks[61][:2],   # mouth left
            landmarks[291][:2],  # mouth right
            landmarks[199][:2]   # chin
        ], dtype="double")

        # 3D model points (tỷ lệ approx, mm)
        model_points = np.array([
            [0.0, 0.0, 0.0],        # Nose tip
            [-30.0, 50.0, -30.0],   # left eye corner
            [30.0, 50.0, -30.0],    # right eye corner
            [-40.0, -50.0, -30.0],  # mouth left
            [40.0, -50.0, -30.0],   # mouth right
            [0.0, -70.0, -20.0]     # chin
        ])

        # Camera internals
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))  # assume no lens distortion

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # chuyển sang góc Euler (deg)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        pitch = np.arctan2(-rmat[2,0], sy)
        yaw = np.arctan2(rmat[1,0], rmat[0,0])
        roll = np.arctan2(rmat[2,1], rmat[2,2])
        return np.array([np.degrees(pitch), np.degrees(yaw), np.degrees(roll)])
