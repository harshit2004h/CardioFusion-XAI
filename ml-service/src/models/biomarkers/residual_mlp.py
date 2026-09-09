import torch
import torch.nn as nn

from .multitask_heads import (
    BiomarkerMultiTaskHeads
)


class ResidualBlock(nn.Module):

    def __init__(
        self,
        dimension: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(
                dimension,
                dimension
            ),
            nn.LayerNorm(
                dimension
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return (
            x
            + self.block(x)
        )


class ResidualMLP(nn.Module):
    """
    Simple residual MLP baseline for
    one tabular source.
    """

    def __init__(
        self,
        input_dim: int,
        target_specs: dict,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
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
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(
                hidden_dim,
                dropout
            )
            for _ in range(num_blocks)
        ])

        self.heads = (
            BiomarkerMultiTaskHeads(
                input_dim=hidden_dim,
                target_specs=target_specs,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        representation = (
            self.input_projection(x)
        )

        representation = (
            self.blocks(representation)
        )

        return self.heads(
            representation
        )   