"""
Training script cho ResNet emotion classification model.
Train trên data/raw/images/train và validate trên data/raw/images/test
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DIR, CHECKPOINTS_DIR, DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS,
    NUM_WORKERS, EMOTION_CLASSES, EMOTION_INPUT_SIZE, EMB_DIM
)
from src.models.emotion_backbone import EmotionResNet
from src.models.trainer import Trainer
from src.data_pipeline.dataset import EmotionDataset

def get_image_paths_and_labels(data_dir):
    """
    Load tất cả image paths và labels từ thư mục data/raw/images/train hoặc test
    """
    data_path = Path(data_dir)
    img_paths = []
    labels = []
    
    for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
        emotion_dir = data_path / emotion
        if not emotion_dir.exists():
            print(f"[WARNING] Directory {emotion_dir} does not exist")
            continue
        
        # Lấy tất cả images trong thư mục emotion
        image_files = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
        
        for img_path in image_files:
            img_paths.append(str(img_path))
            labels.append(emotion_idx)
    
    return img_paths, labels

def main():
    print("[INFO] Starting ResNet training...")
    
    # Load training data
    train_dir = RAW_DIR / "images" / "train"
    train_paths, train_labels = get_image_paths_and_labels(train_dir)
    print(f"[INFO] Found {len(train_paths)} training images")
    
    # Load test data
    test_dir = RAW_DIR / "images" / "test"
    test_paths, test_labels = get_image_paths_and_labels(test_dir)
    print(f"[INFO] Found {len(test_paths)} test images")
    
    if len(train_paths) == 0:
        print("[ERROR] No training images found!")
        return
    
    # Data transforms với augmentation cho training
    train_transform = transforms.Compose([
        transforms.Resize((EMOTION_INPUT_SIZE + 32, EMOTION_INPUT_SIZE + 32)),
        transforms.RandomCrop(EMOTION_INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((EMOTION_INPUT_SIZE, EMOTION_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = EmotionDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = EmotionDataset(test_paths, test_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    # Create model
    model = EmotionResNet(
        num_classes=len(EMOTION_CLASSES),
        embedding_dim=EMB_DIM,
        backbone='resnet18',
        freeze_backbone=False  # Fine-tune toàn bộ
    )
    
    print(f"[INFO] Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(DEVICE),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=str(CHECKPOINTS_DIR),
        early_stop_patience=10
    )
    
    # Train
    trainer.train(
        num_epochs=EPOCHS,
        stage="resnet",
        save_name="resnet_emotion",
        resume_from_checkpoint=None
    )
    
    print("[INFO] Training completed!")

if __name__ == "__main__":
    main()
