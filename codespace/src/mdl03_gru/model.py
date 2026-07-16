import torch
from torch import nn


class GRUFlowClassifier(nn.Module):
    def __init__(
            self,
            num_features: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.1,
            bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        output_size = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape from the shared trainer:
        # [batch_size, num_features]
        #
        # For this GRU implementation, one flow is represented as
        # a sequence of feature values:
        # [batch_size, num_features, 1]
        x = x.unsqueeze(-1)

        _, hidden = self.gru(x)

        if self.bidirectional:
            # Last layer has two directions: forward and backward.
            last_hidden = hidden[-2:].transpose(0, 1).reshape(x.size(0), -1)
        else:
            last_hidden = hidden[-1]

        logits = self.classifier(last_hidden).squeeze(-1)

        return logits


def build_gru_classifier(
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
) -> GRUFlowClassifier:
    return GRUFlowClassifier(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
