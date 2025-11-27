# src/config.py
from pathlib import Path

# =============================================================================
# ĐƯỜNG DẪN CHUNG
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]          # face-emotion-detector/
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = DATA_DIR / "datasets"

EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"

# Tạo thư mục nếu chưa tồn tại
for p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, DATASET_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FACE DETECTION (MediaPipe)
# =============================================================================
FACE_DET_CONFIDENCE = 0.7                     # <-- cái bạn đang thiếu trước đó

# =============================================================================
# FACE MESH (MediaPipe)
# =============================================================================
FACE_MESH_MAX_NUM_FACES = 2
FACE_MESH_REFINE_LANDMARKS = True
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# =============================================================================
# EMOTION CLASSIFICATION
# =============================================================================
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_EMOTION_CLASSES = len(EMOTION_CLASSES)
EMOTION_INPUT_SIZE = 224                       # ResNet, EfficientNet, v.v.

# =============================================================================
# TEMPORAL / ENGAGEMENT MODEL (chuỗi video)
# =============================================================================
SEQ_LEN = 16          # số frame trong 1 sequence
EMB_DIM = 256         # dimension của embedding sau backbone (ResNet)
LANDMARK_DIM = 30     # số đặc trưng landmark bạn trích (ví dụ: EAR, MAR, gaze vector, head pose...)
GRU_OUTPUT_DIM = 2    # focus (1) / unfocus (0)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
NUM_WORKERS = 4
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# =============================================================================
# INFERENCE / REALTIME
# =============================================================================
WEBCAM_INDEX = 0
SHOW_FPS = True
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720