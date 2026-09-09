import torch
import torch.nn as nn


class BinaryClassificationHead(nn.Module):
    """
    Binary classification head.

    Returns logits.
    Sigmoid should be applied during inference/evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            self.head = nn.Linear(
                input_dim,
                1
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    hidden_dim,
                    1
                ),
            )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.head(x)


class MulticlassClassificationHead(nn.Module):
    """
    Multiclass classification head.

    Returns logits.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be >= 2."
            )

        self.head = nn.Linear(
            input_dim,
            num_classes
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.head(x)


class RegressionHead(nn.Module):
    """
    Single-output regression head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            self.head = nn.Linear(
                input_dim,
                1
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    hidden_dim,
                    1
                ),
            )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.head(x)