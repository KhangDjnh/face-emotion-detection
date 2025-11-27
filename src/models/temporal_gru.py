import torch
import torch.nn as nn

class TemporalGRU(nn.Module):
    """
    Temporal GRU model để dự đoán engagement:
    Input: sequence_tensor (batch, seq_len, feature_dim)
        feature_dim = emotion_embedding_dim (từ ResNet)
    Output: logits (batch, 2) -> focus (1) / unfocus (0)
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=1, output_dim=2, dropout=0.2):
        super(TemporalGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classifier head
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: tensor shape (batch, seq_len, feature_dim)
        output: logits (batch, output_dim)
        """
        batch_size = x.size(0)

        # GRU forward
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden_dim)
        # Lấy output cuối cùng trong sequence
        out_last = out[:, -1, :]  # (batch, hidden_dim)

        logits = self.fc(out_last)  # (batch, output_dim)
        return logits
