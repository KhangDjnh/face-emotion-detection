# Face Emotion & Focus Detection Pipeline

Dự án AI nhận diện cảm xúc và mức độ tập trung từ video real-time sử dụng:
1. **Face Detection** - MediaPipe
2. **Face Landmarks** - MediaPipe Face Mesh
3. **Emotion Classification** - ResNet (trained trên images)
4. **Temporal Model** - GRU (trained trên video sequences)

## Cấu trúc Pipeline

```
Video → Frame → Face Detection (MediaPipe) 
              → Face Landmarks (MediaPipe Face Mesh)
              → Emotion Classification (ResNet) → Embedding (256 dims)
              → Temporal Model (GRU) → Focus/Unfocus
```

## Output Format

Mỗi frame/video sẽ có output JSON:

```json
{
  "timestamp": 5.42,
  "emotion": "happy",
  "emotion_conf": 0.92,
  "focus": 1,
  "focus_conf": 0.83
}
```

Trong đó:
- `timestamp`: thời gian (giây)
- `emotion`: một trong 7 cảm xúc: "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
- `emotion_conf`: confidence của emotion (0-1)
- `focus`: 0 = unfocus, 1 = focus
- `focus_conf`: confidence của focus prediction (0-1)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Training Pipeline

### Bước 1: Train ResNet cho Emotion Classification

Train ResNet trên dataset images:

```bash
python scripts/train_resnet.py
```

Model sẽ được lưu tại: `experiments/checkpoints/resnet_emotion_best.pth`

**Dataset structure:**
```
data/raw/images/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── neutral/
  │   ├── sad/
  │   └── surprise/
  └── test/
      ├── angry/
      ├── disgust/
      ├── fear/
      ├── happy/
      ├── neutral/
      ├── sad/
      └── surprise/
```

### Bước 2: Prepare Data cho GRU Training

Extract embeddings từ videos sử dụng ResNet đã train:

```bash
python scripts/prepare_data.py
```

Script này sẽ:
- Load ResNet model đã train
- Process tất cả videos trong `data/raw/videos/focus/` và `data/raw/videos/unfocus/`
- Extract embeddings từ mỗi frame có face
- Tạo sequences với độ dài `SEQ_LEN` (16 frames)
- Lưu sequences tại `data/processed/sequences/sequences.npy` và `labels.npy`

**Video structure:**
```
data/raw/videos/
  ├── focus/
  │   ├── focus_01.mp4
  │   ├── focus_02.mp4
  │   └── ...
  └── unfocus/
      ├── unfocus_01.mp4
      ├── unfocus_02.mp4
      └── ...
```

### Bước 3: Train GRU cho Focus Detection

Train GRU trên sequences đã prepare:

```bash
python scripts/train_gru.py
```

Model sẽ được lưu tại: `experiments/checkpoints/temporal_gru_best.pth`

## Inference

### Real-time từ Webcam

```bash
python run_inference.py
```

hoặc

```bash
python src/inference/pipeline.py
```

### Từ Video File

```bash
python run_inference.py video.mp4
```

### Lưu kết quả ra JSON

```bash
python run_inference.py video.mp4 output.json
```

## Cấu trúc Project

```
face-emotion-detector/
├── data/
│   ├── raw/
│   │   ├── images/          # Training images cho ResNet
│   │   └── videos/          # Training videos cho GRU
│   └── processed/
│       └── sequences/       # Processed sequences cho GRU
├── experiments/
│   └── checkpoints/         # Saved models
├── src/
│   ├── config.py            # Configuration
│   ├── detection/
│   │   └── face_detector.py # MediaPipe face detection
│   ├── landmarks/
│   │   └── face_mesh.py     # MediaPipe face mesh
│   ├── models/
│   │   ├── emotion_backbone.py  # ResNet model
│   │   ├── temporal_gru.py      # GRU model
│   │   └── trainer.py           # Training utilities
│   ├── data_pipeline/
│   │   ├── dataset.py       # Dataset classes
│   │   └── features.py      # Feature extraction
│   ├── inference/
│   │   └── pipeline.py       # Inference pipeline
│   └── utils/
│       ├── io.py            # I/O utilities
│       └── vis.py           # Visualization
├── scripts/
│   ├── train_resnet.py      # Train ResNet
│   ├── train_gru.py         # Train GRU
│   ├── prepare_data.py      # Prepare video sequences
│   └── evaluate.py          # Evaluation scripts
└── run_inference.py         # Main inference script
```

## Configuration

Các tham số chính trong `src/config.py`:

- `SEQ_LEN = 16`: Số frames trong một sequence cho GRU
- `EMB_DIM = 256`: Dimension của embedding từ ResNet
- `GRU_OUTPUT_DIM = 2`: Binary classification (focus/unfocus)
- `EMOTION_INPUT_SIZE = 224`: Input size cho ResNet
- `BATCH_SIZE = 8`: Batch size cho training
- `LR = 1e-4`: Learning rate
- `EPOCHS = 30`: Số epochs

## Model Architecture

### ResNet Emotion Backbone
- Backbone: ResNet18 (pretrained)
- Output: 
  - Emotion logits (7 classes)
  - Embedding (256 dims)

### Temporal GRU
- Input: Sequence of embeddings (seq_len=16, embedding_dim=256)
- Architecture: 2-layer GRU với hidden_dim=128
- Output: Binary classification (focus/unfocus)

## Notes

- ResNet được train trên images để nhận diện emotion
- GRU được train trên video sequences để nhận diện focus/unfocus
- Pipeline inference sử dụng cả 2 models: ResNet cho emotion, GRU cho focus
- Output format JSON với timestamp, emotion, emotion_conf, focus, focus_conf

