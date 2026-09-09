import torch
import torch.nn as nn


class SourceInputAdapter(nn.Module):
    """
    Converts source-specific tabular features into
    feature tokens in a shared embedding space.

    Input:
        x = [batch, num_features]

    Output:
        tokens = [batch, num_features, d_token]
    """

    def __init__(
        self,
        num_features: int,
        d_token: int = 64,
    ):
        super().__init__()

        if num_features <= 0:
            raise ValueError(
                "num_features must be > 0."
            )

        if d_token <= 0:
            raise ValueError(
                "d_token must be > 0."
            )

        self.num_features = num_features
        self.d_token = d_token

        self.weight = nn.Parameter(
            torch.randn(
                num_features,
                d_token
            ) * 0.02
        )

        self.bias = nn.Parameter(
            torch.zeros(
                num_features,
                d_token
            )
        )

        self.norm = nn.LayerNorm(
            d_token
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.ndim != 2:
            raise ValueError(
                f"Expected [batch, features], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} "
                f"features, got {x.shape[1]}."
            )

        tokens = (
            x.unsqueeze(-1)
            * self.weight.unsqueeze(0)
            + self.bias.unsqueeze(0)
        )

        tokens = self.norm(
            tokens
        )

        return tokens