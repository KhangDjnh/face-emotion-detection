"""
Feature extraction functions từ landmarks.
Các functions này được sử dụng trong FaceMeshExtractor, không cần gọi riêng.
"""
import numpy as np

# ============================
# 1) EAR (Eye Aspect Ratio)
# ============================

def compute_EAR(eye_landmarks):
    """
    EAR - Eye Aspect Ratio
    Tính từ 6 landmark points của mắt.
    
    Args:
        eye_landmarks: np.array shape (6, 2) hoặc (6, 3) - 6 điểm mắt
    
    Returns:
        float: EAR value
    """
    if len(eye_landmarks) < 6:
        return 0.0
    
    # Lấy 6 điểm: [p0, p1, p2, p3, p4, p5]
    # EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    p0 = eye_landmarks[0][:2]  # chỉ lấy x, y
    p1 = eye_landmarks[1][:2]
    p2 = eye_landmarks[2][:2]
    p3 = eye_landmarks[3][:2]
    p4 = eye_landmarks[4][:2]
    p5 = eye_landmarks[5][:2]
    
    # Tính distances
    vertical_1 = np.linalg.norm(p1 - p5)
    vertical_2 = np.linalg.norm(p2 - p4)
    horizontal = np.linalg.norm(p0 - p3)
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
    return ear

# ============================
# 2) MAR (Mouth Aspect Ratio)
# ============================

def compute_MAR(mouth_landmarks):
    """
    MAR - Mouth Aspect Ratio
    Tính từ các landmark points của miệng.
    
    Args:
        mouth_landmarks: np.array shape (N, 2) hoặc (N, 3) - các điểm môi
    
    Returns:
        float: MAR value
    """
    if len(mouth_landmarks) < 4:
        return 0.0
    
    # Simplified MAR: sử dụng 4 điểm chính
    # Có thể mở rộng với nhiều điểm hơn
    if len(mouth_landmarks) >= 8:
        # Sử dụng 8 điểm
        p1 = mouth_landmarks[0][:2]
        p2 = mouth_landmarks[1][:2]
        p3 = mouth_landmarks[2][:2]
        p4 = mouth_landmarks[3][:2]
        p5 = mouth_landmarks[4][:2]
        p6 = mouth_landmarks[5][:2]
        p7 = mouth_landmarks[6][:2]
        p8 = mouth_landmarks[7][:2]
        
        # Vertical distances
        vert_1 = np.linalg.norm(p3 - p7)
        vert_2 = np.linalg.norm(p4 - p8)
        # Horizontal distance
        horz = np.linalg.norm(p1 - p2)
        
        mar = (vert_1 + vert_2) / (2.0 * horz + 1e-6)
    else:
        # Simplified với 4 điểm
        p1 = mouth_landmarks[0][:2]
        p2 = mouth_landmarks[1][:2]
        p3 = mouth_landmarks[2][:2]
        p4 = mouth_landmarks[3][:2]
        
        vert = np.linalg.norm(p2 - p4)
        horz = np.linalg.norm(p1 - p3)
        mar = vert / (horz + 1e-6)
    
    return mar

# ============================
# 3) Head Pose (Yaw / Pitch / Roll)
# ============================

def compute_head_pose_simple(landmarks):
    """
    Simplified head pose estimation từ landmarks.
    Trả về approximate yaw, pitch, roll.
    
    Args:
        landmarks: np.array shape (N, 2) hoặc (N, 3) - các landmark points
    
    Returns:
        tuple: (yaw, pitch, roll) in degrees (approximate)
    """
    if len(landmarks) < 5:
        return 0.0, 0.0, 0.0
    
    # Simplified: sử dụng relative positions
    # Cần landmarks với indices cụ thể (tùy MediaPipe)
    # Ở đây chỉ là placeholder, thực tế sẽ dùng trong FaceMeshExtractor
    
    # Placeholder implementation
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    
    return yaw, pitch, roll

# ============================
# 4) Tổng hợp features (deprecated - dùng trong FaceMeshExtractor)
# ============================

def compute_features(landmarks):
    """
    DEPRECATED: Function này không được dùng trực tiếp.
    FaceMeshExtractor đã có các methods riêng.
    
    Giữ lại để backward compatibility.
    """
    # MediaPipe Face Mesh có 468 landmarks
    # Các indices cho eyes và mouth
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [61, 291, 81, 311, 78, 308, 13, 14]
    
    if len(landmarks) < 468:
        return np.zeros(6, dtype=np.float32)
    
    # Extract eye và mouth landmarks
    left_eye_pts = landmarks[LEFT_EYE_INDICES]
    right_eye_pts = landmarks[RIGHT_EYE_INDICES]
    mouth_pts = landmarks[MOUTH_INDICES]
    
    # Compute features
    EAR_left = compute_EAR(left_eye_pts)
    EAR_right = compute_EAR(right_eye_pts)
    MAR = compute_MAR(mouth_pts)
    yaw, pitch, roll = compute_head_pose_simple(landmarks)
    
    # Feature vector
    features = np.array([
        EAR_left,
        EAR_right,
        MAR,
        yaw,
        pitch,
        roll
    ], dtype=np.float32)
    
    return features
