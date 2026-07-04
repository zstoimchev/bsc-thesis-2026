import torch
from torch import nn


class MLPClassifierTorch(nn.Module):
    def __init__(
            self,
            num_features: int,
            hidden_layer_sizes: tuple[int, ...] = (64, 32),
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        input_size = num_features

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output shape: [batch_size]
        return self.network(x).squeeze(-1)


def build_mlp_classifier(
        num_features: int,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
) -> MLPClassifierTorch:
    return MLPClassifierTorch(
        num_features=num_features,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout,
    )
