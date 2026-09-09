import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Simple MLP block:

        Linear
        LayerNorm
        GELU
        Dropout
        Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be > 0."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be > 0."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be > 0."
            )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                output_dim
            ),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.network(x)