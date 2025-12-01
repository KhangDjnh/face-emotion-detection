# src/models/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import os
import time
from typing import Optional
from tqdm import tqdm

class Trainer:
    """
    Trainer class cho ResNet (stage1) và Temporal GRU (stage2) 
    - Hỗ trợ checkpoint per epoch
    - Resume training
    - Early stopping
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = torch.device('cpu'),
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "checkpoints",
        early_stop_patience: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.early_stop_patience = early_stop_patience

    def train_epoch(self, epoch=None, total_epochs=None):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Progress bar cho training
        epoch_info = f"Epoch {epoch}/{total_epochs}" if epoch and total_epochs else "Training"
        pbar = tqdm(
            self.train_loader,
            desc=f"[TRAIN] {epoch_info}",
            unit="batch",
            leave=False,
            ncols=100
        )
        
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            # Kiểm tra model type
            if 'TemporalGRU' in type(self.model).__name__:
                logits = self.model(X)
            else:
                # ResNet returns (logits, embedding)
                logits, _ = self.model(X)
            
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Update progress bar
            current_loss = running_loss / ((batch_idx + 1) * X.size(0))
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'batch': f'{batch_idx+1}/{len(self.train_loader)}'
            })
        
        pbar.close()
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return epoch_loss, acc, f1

    @torch.no_grad()
    def validate_epoch(self, epoch=None, total_epochs=None):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Progress bar cho validation
        epoch_info = f"Epoch {epoch}/{total_epochs}" if epoch and total_epochs else "Validating"
        pbar = tqdm(
            self.val_loader,
            desc=f"[VAL]   {epoch_info}",
            unit="batch",
            leave=False,
            ncols=100
        )
        
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)
            
            if 'TemporalGRU' in type(self.model).__name__:
                logits = self.model(X)
            else:
                # ResNet returns (logits, embedding)
                logits, _ = self.model(X)
            
            loss = self.criterion(logits, y)
            running_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Update progress bar
            current_loss = running_loss / ((batch_idx + 1) * X.size(0))
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'batch': f'{batch_idx+1}/{len(self.val_loader)}'
            })
        
        pbar.close()

        epoch_loss = running_loss / len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return epoch_loss, acc, f1

    def train(
        self,
        num_epochs: int = 50,
        stage: str = "gru",
        save_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        best_val_f1 = 0.0
        patience_counter = 0
        start_epoch = 0

        # Resume training nếu có checkpoint
        if resume_from_checkpoint:
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[INFO] Resuming training from epoch {start_epoch}")

        print(f"\n{'='*80}")
        print(f"[{stage.upper()}] Starting training for {num_epochs} epochs")
        print(f"[{stage.upper()}] Device: {self.device}")
        print(f"[{stage.upper()}] Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"{'='*80}\n")
        
        epoch_times = []
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            print(f"\n[{stage.upper()}] {'='*70}")
            print(f"[{stage.upper()}] Epoch {epoch+1}/{num_epochs}")
            print(f"[{stage.upper()}] {'='*70}")
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(epoch+1, num_epochs)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self.validate_epoch(epoch+1, num_epochs)
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            avg_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_time = avg_time * remaining_epochs
            
            # Print epoch summary
            print(f"\n[{stage.upper()}] Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
            print(f"  Time:  {epoch_time:.2f}s | Avg: {avg_time:.2f}s | Est. remaining: {estimated_time/60:.1f}min")

            # Save checkpoint per epoch
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"{save_name}_{stage}_epoch{epoch+1}.pth" if save_name else f"{stage}_epoch{epoch+1}.pth"
            )
            print(f"[{stage.upper()}] Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"[{stage.upper()}] Checkpoint saved!")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_path = os.path.join(
                    self.checkpoint_dir,
                    f"{save_name}_{stage}_best.pth" if save_name else f"{stage}_best.pth"
                )
                print(f"[{stage.upper()}] New best F1 score: {best_val_f1:.4f}!")
                print(f"[{stage.upper()}] Saving best model to {best_path}...")
                torch.save(self.model.state_dict(), best_path)
                print(f"[{stage.upper()}] Best model saved!")
            else:
                patience_counter += 1
                print(f"[{stage.upper()}] No improvement. Patience: {patience_counter}/{self.early_stop_patience}")
                if patience_counter >= self.early_stop_patience:
                    print(f"\n[{stage.upper()}] Early stopping triggered at epoch {epoch+1}")
                    print(f"[{stage.upper()}] Best validation F1: {best_val_f1:.4f}")
                    break
        
        print(f"\n{'='*80}")
        print(f"[{stage.upper()}] Training completed!")
        print(f"[{stage.upper()}] Best validation F1: {best_val_f1:.4f}")
        print(f"[{stage.upper()}] Total training time: {sum(epoch_times)/60:.2f} minutes")
        print(f"{'='*80}\n")
