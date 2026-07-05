import torch
from torch import nn


class TabularTransformer(nn.Module):
    def __init__(
            self,
            num_features: int,
            d_model: int = 32,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.value_projection = nn.Linear(1, d_model)

        self.feature_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_features]
        x = x.unsqueeze(-1)

        # tokens shape: [batch_size, num_features, d_model]
        tokens = self.value_projection(x)
        tokens = tokens + self.feature_embedding

        encoded = self.encoder(tokens)

        # average pooling over feature tokens
        pooled = encoded.mean(dim=1)

        # output shape: [batch_size]
        logits = self.classifier(pooled).squeeze(-1)

        return logits


def build_tabular_transformer(
        num_features: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
) -> TabularTransformer:
    return TabularTransformer(
        num_features=num_features,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
