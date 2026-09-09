import torch
import torch.nn as nn


class FeatureLayerNorm(nn.Module):
    """
    LayerNorm wrapper for feature/token representations.

    Expected input:
        [batch, features, embedding_dim]
        or
        [batch, embedding_dim]
    """

    def __init__(self, dimension: int):
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        self.norm = nn.LayerNorm(
            dimension
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.norm(x)