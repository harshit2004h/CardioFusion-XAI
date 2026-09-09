import torch
import torch.nn as nn

from torchvision.models.video import (
    r2plus1d_18,
    R2Plus1D_18_Weights,
)

from .regression_heads import (
    EFRegressionHead
)


class R2Plus1DRegressor(
    nn.Module
):
    """
    R(2+1)D-18 video model for
    EchoNet-Dynamic EF regression.

    Input:
        [batch, 3, 32, 112, 112]

    Output:
        [batch, 1]
    """

    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = (
            R2Plus1D_18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = r2plus1d_18(
            weights=weights
        )

        feature_dim = (
            self.backbone.fc.in_features
        )

        # Remove torchvision's
        # classification head.
        self.backbone.fc = nn.Identity()

        self.regression_head = (
            EFRegressionHead(
                input_dim=feature_dim
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "Expected video input "
                "[batch, channels, frames, "
                f"height, width], got {tuple(x.shape)}."
            )

        features = self.backbone(
            x
        )

        return self.regression_head(
            features
        )