"""
Dataset classes cho training ResNet và GRU.
"""
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from torchvision import transforms
import numpy as np

# ---------------- Frame-level dataset ---------------- 
class EmotionDataset(Dataset):
    """
    Dataset cho train ResNet (frame-level).
    Load images từ thư mục data/raw/images/train hoặc test.
    """

    def __init__(
        self,
        img_paths: List[str | Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.img_paths = [str(p) for p in img_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Cannot read image: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)  # chuyển sang PIL

            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        except Exception as e:
            print(f"[ERROR] Error loading image {img_path}: {e}")
            # Return a dummy image if error
            dummy_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_img = self.transform(dummy_img)
            else:
                dummy_img = transforms.ToTensor()(dummy_img)
            return dummy_img, torch.tensor(0, dtype=torch.long)

# ---------------- Sequence-level dataset ---------------- 
class EmotionSequenceDataset(Dataset):
    """
    Dataset cho GRU:
    - Mỗi sample là sequence gồm N frames
    - Mỗi frame: emotion embedding (từ ResNet)
    """

    def __init__(
        self,
        sequences: List[np.ndarray] | np.ndarray,
        labels: List[int] | np.ndarray
    ):
        """
        sequences: List of sequences hoặc np.array
            sequences[i] = np.array shape (seq_len, embedding_dim)
        labels: List[int] hoặc np.ndarray engagement labels (0=unfocus, 1=focus)
        """
        if isinstance(sequences, list):
            self.sequences = sequences
        else:
            # Convert np.array to list of arrays
            self.sequences = [sequences[i] for i in range(len(sequences))]
        
        if isinstance(labels, np.ndarray):
            self.labels = labels.tolist()
        else:
            self.labels = labels
        
        assert len(self.sequences) == len(self.labels), \
            f"Sequences ({len(self.sequences)}) and labels ({len(self.labels)}) must have same length"

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # Convert np array -> tensor
        if isinstance(seq, np.ndarray):
            seq_tensor = torch.tensor(seq, dtype=torch.float32)  # (seq_len, embedding_dim)
        else:
            seq_tensor = torch.tensor(np.array(seq), dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq_tensor, label
