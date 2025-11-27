# src/models/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import os
from typing import Optional

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

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for X, y in self.train_loader:
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

        epoch_loss = running_loss / len(self.train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return epoch_loss, acc, f1

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for X, y in self.val_loader:
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

        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate_epoch()

            print(f"[{stage.upper()}] Epoch {epoch+1}/{num_epochs} | "
                  f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f} | "
                  f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}")

            # Save checkpoint per epoch
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"{save_name}_{stage}_epoch{epoch+1}.pth" if save_name else f"{stage}_epoch{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_path = os.path.join(
                    self.checkpoint_dir,
                    f"{save_name}_{stage}_best.pth" if save_name else f"{stage}_best.pth"
                )
                torch.save(self.model.state_dict(), best_path)
                print(f"[INFO] Saved best model to {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    print(f"[INFO] Early stopping at epoch {epoch+1}")
                    break
