import torch
import torch.nn as nn


class EFRegressionHead(
    nn.Module
):
    """
    Single-output regression head
    for LVEF prediction.
    """

    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.linear(x)