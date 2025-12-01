# scripts/evaluate.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.models.emotion_backbone import EmotionResNet
from src.models.temporal_gru import TemporalGRU
from src.data_pipeline.dataset import EmotionDataset, EmotionSequenceDataset
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def evaluate_resnet(model_path, test_paths, test_labels):
    test_dataset = EmotionDataset(test_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = EmotionResNet(num_classes=7, embedding_dim=256, backbone='resnet18', freeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(imgs)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    print("Accuracy:", accuracy_score(trues, preds))
    print("F1-score:", f1_score(trues, preds, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(trues, preds))

def evaluate_gru(model_path, seq_test, label_test):
    test_dataset = EmotionSequenceDataset(sequences=seq_test, labels=label_test)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    feature_dim = seq_test[0].shape[1]
    model = TemporalGRU(input_dim=feature_dim, hidden_dim=128, output_dim=3)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    preds, trues = [], []
    with torch.no_grad():
        for X_seq, y in test_loader:
            X_seq, y = X_seq.to(DEVICE), y.to(DEVICE)
            logits = model(X_seq)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())

    print("Accuracy:", accuracy_score(trues, preds))
    print("F1-score:", f1_score(trues, preds, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(trues, preds))

if __name__ == "__main__":
    # TODO: provide test dataset paths and labels
    # evaluate_resnet("experiments/resnet_emotion_best.pth", test_paths, test_labels)
    # evaluate_gru("experiments/temporal_gru_best.pth", seq_test, label_test)
    print("Modify __main__ to provide test dataset paths and labels.")
