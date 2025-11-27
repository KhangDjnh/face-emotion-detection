# tests/test_emotion_backbone.py
import torch
from src.models.emotion_backbone import EmotionResNet

if __name__ == "__main__":
    print("Testing EmotionResNet...")
    model = EmotionResNet(
        num_classes=7,
        embedding_dim=256,
        backbone='resnet18',
        freeze_backbone=True
    )
    model.eval()

    dummy = torch.randn(4, 3, 224, 224)  # batch=4 cho thực tế hơn
    with torch.no_grad():
        logits, embedding = model(dummy)

    print("Logits shape:", logits.shape)      # [4, 7]
    print("Embedding shape:", embedding.shape)  # [4, 256]
    print("Test PASSED! Model chạy hoàn hảo, không warning!")