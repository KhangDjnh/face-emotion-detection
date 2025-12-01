"""
Training script cho Temporal GRU model.
Train trên sequences đã được prepare từ videos.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY,
    EPOCHS, NUM_WORKERS, SEQ_LEN, EMB_DIM, GRU_OUTPUT_DIM
)
from src.models.temporal_gru import TemporalGRU
from src.models.trainer import Trainer

def main():
    print("\n" + "="*80)
    print(" " * 25 + "TEMPORAL GRU TRAINING")
    print("="*80 + "\n")
    
    print("[STEP 1/6] Loading sequences and labels...")
    sequences_dir = PROCESSED_DIR / "sequences"
    sequences_file = sequences_dir / "sequences.npy"
    labels_file = sequences_dir / "labels.npy"
    
    if not sequences_file.exists() or not labels_file.exists():
        print("[ERROR] Sequences not found!")
        print("[INFO] Please run prepare_data.py first to create sequences")
        return
    
    sequences = np.load(sequences_file)  # (num_sequences, seq_len, EMB_DIM)
    labels = np.load(labels_file)  # (num_sequences,)
    
    print(f"  ✓ Loaded {len(sequences)} sequences")
    print(f"  ✓ Sequences shape: {sequences.shape}")
    print(f"  ✓ Sequence length: {SEQ_LEN} frames")
    print(f"  ✓ Embedding dimension: {EMB_DIM}")
    focus_count = np.sum(labels)
    unfocus_count = len(labels) - focus_count
    print(f"  ✓ Labels distribution: Focus={focus_count} ({focus_count/len(labels)*100:.1f}%), Unfocus={unfocus_count} ({unfocus_count/len(labels)*100:.1f}%)")
    
    print("\n[STEP 2/6] Splitting train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"  ✓ Train sequences: {len(X_train)}")
    print(f"  ✓ Val sequences: {len(X_val)}")
    print(f"  ✓ Train split: {len(X_train)/len(sequences)*100:.1f}%")
    print(f"  ✓ Val split: {len(X_val)/len(sequences)*100:.1f}%")
    
    print("\n[STEP 3/6] Converting to tensors...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    print(f"  ✓ Train tensor shape: {X_train_tensor.shape}")
    print(f"  ✓ Val tensor shape: {X_val_tensor.shape}")
    
    print("\n[STEP 4/6] Creating datasets and data loaders...")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
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
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    
    print("\n[STEP 5/6] Creating model...")
    model = TemporalGRU(
        input_dim=EMB_DIM,
        hidden_dim=128,
        num_layers=2,
        output_dim=GRU_OUTPUT_DIM,
        dropout=0.3
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model: Temporal GRU")
    print(f"  ✓ Input dimension: {EMB_DIM}")
    print(f"  ✓ Hidden dimension: 128")
    print(f"  ✓ Number of layers: 2")
    print(f"  ✓ Dropout: 0.3")
    print(f"  ✓ Output classes: {GRU_OUTPUT_DIM} (Focus/Unfocus)")
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    
    print("\n[STEP 6/6] Initializing trainer...")
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
    print(f"  ✓ Learning rate: {LR}")
    print(f"  ✓ Weight decay: {WEIGHT_DECAY}")
    print(f"  ✓ Batch size: {BATCH_SIZE}")
    print(f"  ✓ Epochs: {EPOCHS}")
    print(f"  ✓ Early stop patience: 10")
    print(f"  ✓ Checkpoint directory: {CHECKPOINTS_DIR}")
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Train
    trainer.train(
        num_epochs=EPOCHS,
        stage="gru",
        save_name="temporal_gru",
        resume_from_checkpoint=None
    )
    
    print("\n" + "="*80)
    print(" " * 30 + "TRAINING COMPLETED!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
