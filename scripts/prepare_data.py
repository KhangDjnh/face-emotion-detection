"""
Prepare data từ videos: extract frames, detect faces, extract landmarks và embeddings.
Tạo sequences cho GRU training.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from src.config import (
    RAW_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE, SEQ_LEN, EMB_DIM,
    EMOTION_INPUT_SIZE, EMOTION_CLASSES
)
from src.detection.face_detector import FaceDetector
from src.landmarks.face_mesh import FaceMeshExtractor
from src.models.emotion_backbone import EmotionResNet

# Image transform cho ResNet
transform = transforms.Compose([
    transforms.Resize((EMOTION_INPUT_SIZE, EMOTION_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_embedding_from_face(face_crop, resnet_model):
    """
    Extract embedding từ face crop sử dụng ResNet.
    """
    if face_crop is None or face_crop.size == 0:
        return None
    
    try:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        tensor_face = transform(face_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _, embedding = resnet_model(tensor_face)
            embedding = embedding.squeeze(0).cpu().numpy()  # (EMB_DIM,)
        
        return embedding
    except Exception as e:
        print(f"[WARNING] Error extracting embedding: {e}")
        return None

def process_video(video_path, face_detector, face_mesh, resnet_model, label):
    """
    Process một video: extract embeddings từ mỗi frame có face.
    Returns: list of embeddings (mỗi embedding là np.array shape (EMB_DIM,))
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []
    
    embeddings = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect face
        boxes, scores = face_detector.detect(frame)
        if len(boxes) == 0:
            continue
        
        # Lấy face đầu tiên
        x1, y1, x2, y2 = boxes[0]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        face_crop = frame[y1:y2, x1:x2]
        
        # Extract embedding từ ResNet
        embedding = extract_embedding_from_face(face_crop, resnet_model)
        if embedding is not None:
            embeddings.append(embedding)
    
    cap.release()
    return embeddings

def create_sequences(embeddings, seq_len=SEQ_LEN):
    """
    Tạo sequences từ list of embeddings.
    Mỗi sequence có độ dài seq_len.
    """
    sequences = []
    if len(embeddings) < seq_len:
        # Nếu không đủ frames, pad với frame cuối cùng
        padded = embeddings + [embeddings[-1]] * (seq_len - len(embeddings))
        sequences.append(padded)
    else:
        # Sliding window với stride = 1
        for i in range(len(embeddings) - seq_len + 1):
            sequences.append(embeddings[i:i+seq_len])
    
    return sequences

def main():
    print("[INFO] Preparing data from videos...")
    
    # Load ResNet model để extract embeddings
    print("[INFO] Loading ResNet model...")
    resnet_model = EmotionResNet(
        num_classes=len(EMOTION_CLASSES),
        embedding_dim=EMB_DIM,
        backbone='resnet18',
        freeze_backbone=True
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
        print("[WARNING] Please train ResNet first using train_resnet.py")
        return
    
    resnet_model.to(DEVICE).eval()
    
    # Initialize detectors
    face_detector = FaceDetector()
    face_mesh = FaceMeshExtractor()
    
    # Process videos
    video_dir = RAW_DIR / "videos"
    output_dir = PROCESSED_DIR / "sequences"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_sequences = []
    all_labels = []
    
    # Process focus videos (label = 1)
    focus_dir = video_dir / "focus"
    if focus_dir.exists():
        focus_videos = list(focus_dir.glob("*.mp4"))
        print(f"[INFO] Processing {len(focus_videos)} focus videos...")
        
        for video_path in tqdm(focus_videos, desc="Focus videos"):
            embeddings = process_video(video_path, face_detector, face_mesh, resnet_model, label=1)
            if len(embeddings) > 0:
                sequences = create_sequences(embeddings, SEQ_LEN)
                all_sequences.extend(sequences)
                all_labels.extend([1] * len(sequences))  # 1 = focus
    
    # Process unfocus videos (label = 0)
    unfocus_dir = video_dir / "unfocus"
    if unfocus_dir.exists():
        unfocus_videos = list(unfocus_dir.glob("*.mp4"))
        print(f"[INFO] Processing {len(unfocus_videos)} unfocus videos...")
        
        for video_path in tqdm(unfocus_videos, desc="Unfocus videos"):
            embeddings = process_video(video_path, face_detector, face_mesh, resnet_model, label=0)
            if len(embeddings) > 0:
                sequences = create_sequences(embeddings, SEQ_LEN)
                all_sequences.extend(sequences)
                all_labels.extend([0] * len(sequences))  # 0 = unfocus
    
    if len(all_sequences) == 0:
        print("[ERROR] No sequences created!")
        return
    
    # Save sequences
    print(f"[INFO] Created {len(all_sequences)} sequences")
    print(f"[INFO] Focus: {sum(all_labels)}, Unfocus: {len(all_labels) - sum(all_labels)}")
    
    # Convert to numpy arrays
    sequences_array = np.array(all_sequences)  # (num_sequences, seq_len, EMB_DIM)
    labels_array = np.array(all_labels)
    
    # Save
    sequences_file = output_dir / "sequences.npy"
    labels_file = output_dir / "labels.npy"
    
    np.save(sequences_file, sequences_array)
    np.save(labels_file, labels_array)
    
    print(f"[INFO] Saved sequences to {sequences_file}")
    print(f"[INFO] Saved labels to {labels_file}")
    print(f"[INFO] Sequences shape: {sequences_array.shape}")
    print(f"[INFO] Labels shape: {labels_array.shape}")

if __name__ == "__main__":
    main()
