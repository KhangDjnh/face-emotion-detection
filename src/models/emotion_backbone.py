# src/models/emotion_backbone.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights


class EmotionResNet(nn.Module):
    """
    ResNet backbone cho emotion classification + embedding output.
    ĐÃ LOẠI BỎ HOÀN TOÀN warning pretrained/weights.
    """

    def __init__(self, num_classes=7, embedding_dim=256, backbone='resnet18', freeze_backbone=False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if backbone == 'resnet18' else ResNet34_Weights.DEFAULT

        # Load backbone với weights mới nhất (không warning)
        self.backbone = models.resnet18(weights=weights) if backbone == 'resnet18' \
            else models.resnet34(weights=weights)

        # Freeze backbone nếu cần
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Lấy số feature cuối cùng
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # bỏ fc cũ đi

        # Custom head
        self.embedding_head = nn.Linear(in_features, embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)           # (B, 512)
        embedding = self.relu(self.embedding_head(features))  # (B, embedding_dim)
        logits = self.classifier(embedding)   # (B, num_classes)
        return logits, embedding

    def get_embedding(self, x):
        """Dùng khi chỉ cần embedding (ví dụ faiss search, clustering)"""
        with torch.no_grad():
            features = self.backbone(x)
            embedding = self.relu(self.embedding_head(features))
        return embedding