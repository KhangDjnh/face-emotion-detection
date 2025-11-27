import cv2
import torch
import numpy as np
import json
import time
from collections import deque
from pathlib import Path
from torchvision import transforms
from PIL import Image

from src.config import (
    DEVICE, SEQ_LEN, EMB_DIM, GRU_OUTPUT_DIM,
    EMOTION_CLASSES, EMOTION_INPUT_SIZE,
    CHECKPOINTS_DIR
)
from src.detection.face_detector import FaceDetector
from src.landmarks.face_mesh import FaceMeshExtractor
from src.models.emotion_backbone import EmotionResNet
from src.models.temporal_gru import TemporalGRU

# ---------------- Load models ----------------
print("[INFO] Loading models...")
face_detector = FaceDetector()
face_mesh = FaceMeshExtractor()

# Emotion ResNet
resnet_model = EmotionResNet(
    num_classes=len(EMOTION_CLASSES),
    embedding_dim=EMB_DIM,
    backbone='resnet18',
    freeze_backbone=False
)
resnet_checkpoint = CHECKPOINTS_DIR / "resnet_emotion_best.pth"
if resnet_checkpoint.exists():
    checkpoint = torch.load(resnet_checkpoint, map_location=DEVICE)
    # Handle both dict format and state_dict format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        resnet_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        resnet_model.load_state_dict(checkpoint)
    print(f"[INFO] Loaded ResNet from {resnet_checkpoint}")
else:
    print(f"[WARNING] ResNet checkpoint not found at {resnet_checkpoint}")
resnet_model.to(DEVICE).eval()

# Temporal GRU - chỉ nhận embedding từ ResNet
gru_model = TemporalGRU(
    input_dim=EMB_DIM,
    hidden_dim=128,
    num_layers=1,
    output_dim=GRU_OUTPUT_DIM,
    dropout=0.2
)
gru_checkpoint = CHECKPOINTS_DIR / "temporal_gru_best.pth"
if gru_checkpoint.exists():
    checkpoint = torch.load(gru_checkpoint, map_location=DEVICE)
    # Handle both dict format and state_dict format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gru_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gru_model.load_state_dict(checkpoint)
    print(f"[INFO] Loaded GRU from {gru_checkpoint}")
else:
    print(f"[WARNING] GRU checkpoint not found at {gru_checkpoint}")
gru_model.to(DEVICE).eval()

# ---------------- Image transforms ----------------
transform = transforms.Compose([
    transforms.Resize((EMOTION_INPUT_SIZE, EMOTION_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------- Buffer for sequence ----------------
embedding_buffer = deque(maxlen=SEQ_LEN)

# ---------------- Helper functions ----------------
def extract_emotion_and_embedding(face_crop):
    """
    Extract emotion và embedding từ face crop.
    Returns: (emotion_label, emotion_conf, embedding)
    """
    if face_crop is None or face_crop.size == 0:
        return None, 0.0, None
    
    # Preprocess face crop
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    tensor_face = transform(face_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits, embedding = resnet_model(tensor_face)
        probs = torch.softmax(logits, dim=1)
        emotion_idx = torch.argmax(probs, dim=1).item()
        emotion_conf = probs[0, emotion_idx].item()
        embedding = embedding.squeeze(0).cpu().numpy()  # (EMB_DIM,)
    
    emotion_label = EMOTION_CLASSES[emotion_idx]
    return emotion_label, emotion_conf, embedding

def process_frame(frame, timestamp):
    """
    Process một frame và trả về kết quả JSON.
    Returns: dict với format {
        "timestamp": float,
        "emotion": str,
        "emotion_conf": float,
        "focus": int (0 or 1),
        "focus_conf": float
    } hoặc None nếu không detect được face
    """
    # Face detection
    boxes, scores = face_detector.detect(frame)
    if len(boxes) == 0:
        return None
    
    # Lấy face đầu tiên
    x1, y1, x2, y2 = boxes[0]
    # Đảm bảo coordinates hợp lệ
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    face_crop = frame[y1:y2, x1:x2]
    
    # Extract emotion và embedding
    emotion_label, emotion_conf, embedding = extract_emotion_and_embedding(face_crop)
    
    if embedding is None:
        return None
    
    # Thêm embedding vào buffer
    embedding_buffer.append(embedding)
    
    # Nếu chưa đủ sequence, chỉ trả về emotion
    if len(embedding_buffer) < SEQ_LEN:
        return {
            "timestamp": timestamp,
            "emotion": emotion_label,
            "emotion_conf": emotion_conf,
            "focus": 0,  # chưa đủ data để predict
            "focus_conf": 0.0
        }
    
    # Predict focus với GRU
    seq_tensor = torch.tensor(
        np.stack(list(embedding_buffer), axis=0),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)  # (1, seq_len, EMB_DIM)
    
    with torch.no_grad():
        logits = gru_model(seq_tensor)
        probs = torch.softmax(logits, dim=1)
        focus_idx = torch.argmax(probs, dim=1).item()
        focus_conf = probs[0, focus_idx].item()
    
    return {
        "timestamp": timestamp,
        "emotion": emotion_label,
        "emotion_conf": emotion_conf,
        "focus": focus_idx,  # 0 = unfocus, 1 = focus
        "focus_conf": focus_conf
    }

# ---------------- Main inference function ----------------
def run_realtime_inference(video_source=0, output_file=None):
    """
    Run real-time inference từ webcam hoặc video file.
    
    Args:
        video_source: int (webcam index) hoặc str (video file path)
        output_file: optional, path to save JSON results
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_source}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) if isinstance(video_source, str) else 30
    frame_time = 1.0 / fps if fps > 0 else 0.033
    
    results = []
    frame_count = 0
    start_time = time.time()
    
    print("[INFO] Starting inference... Press 'q' to quit, 'ESC' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time() - start_time
        timestamp = current_time  # hoặc frame_count * frame_time
        
        # Process frame
        result = process_frame(frame, timestamp)
        
        if result:
            results.append(result)
            print(f"[Frame {frame_count}] {json.dumps(result, indent=2)}")
            
            # Visualize trên frame
            emotion_text = f"{result['emotion']} ({result['emotion_conf']:.2f})"
            focus_text = f"Focus: {result['focus']} ({result['focus_conf']:.2f})"
            
            cv2.putText(frame, emotion_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, focus_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Face Emotion & Focus Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' hoặc ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save results nếu có output_file
    if output_file and results:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Results saved to {output_path}")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Có thể truyền video file hoặc dùng webcam
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_realtime_inference(video_source, output_file)
