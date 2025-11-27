"""
Training script cho Temporal GRU model.
Train trên sequences đã được prepare từ videos.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY,
    EPOCHS, NUM_WORKERS, SEQ_LEN, EMB_DIM, GRU_OUTPUT_DIM
)
from src.models.temporal_gru import TemporalGRU
from src.models.trainer import Trainer

def main():
    print("[INFO] Starting GRU training...")
    
    # Load sequences và labels
    sequences_dir = PROCESSED_DIR / "sequences"
    sequences_file = sequences_dir / "sequences.npy"
    labels_file = sequences_dir / "labels.npy"
    
    if not sequences_file.exists() or not labels_file.exists():
        print("[ERROR] Sequences not found!")
        print("[INFO] Please run prepare_data.py first to create sequences")
        return
    
    sequences = np.load(sequences_file)  # (num_sequences, seq_len, EMB_DIM)
    labels = np.load(labels_file)  # (num_sequences,)
    
    print(f"[INFO] Loaded {len(sequences)} sequences")
    print(f"[INFO] Sequences shape: {sequences.shape}")
    print(f"[INFO] Labels distribution: Focus={np.sum(labels)}, Unfocus={len(labels)-np.sum(labels)}")
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
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
    model = TemporalGRU(
        input_dim=EMB_DIM,
        hidden_dim=128,
        num_layers=2,
        output_dim=GRU_OUTPUT_DIM,
        dropout=0.3
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
        stage="gru",
        save_name="temporal_gru",
        resume_from_checkpoint=None
    )
    
    print("[INFO] Training completed!")

if __name__ == "__main__":
    main()
