import torch
import torch.nn as nn

from .multitask_heads import (
    ECGMultiTaskHeads
)


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample=None,
    ):
        super().__init__()

        width = planes

        self.conv1 = nn.Conv1d(
            in_channels,
            width,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(
            width
        )

        self.conv2 = nn.Conv1d(
            width,
            width,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(
            width
        )

        self.conv3 = nn.Conv1d(
            width,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm1d(
            planes * self.expansion
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.downsample = (
            downsample
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# Block-count configurations, keyed by nominal depth.
# NOTE: 101 is the original hardcoded configuration. It is a huge amount of
# capacity (~40M+ params) for the ~15k-record PTB-XL training split used in
# this project and is the primary driver of the reported ECG overfitting
# (train loss keeps falling while val loss plateaus/rises). Prefer 18 or 34
# unless you have verified more data / stronger regularization closes the
# train/val gap.
_DEPTH_CONFIGS: dict[int, list[int]] = {
    18: [1, 1, 1, 1],
    34: [2, 2, 2, 2],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class XResNet1D(
    nn.Module
):
    """
    Deep 1D residual ECG network.

    Designed for:
        PTB-XL 12-lead ECG

    Input:
        [batch, 12, 1000]

    Output:
        {
            "diagnostic": [batch, 5],
            "rhythm": [batch, num_rhythm_classes]
        }
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_diagnostic_classes: int = 5,
        num_rhythm_classes: int = 0,
        depth: int = 18,
        dropout: float = 0.20,
    ):
        super().__init__()

        if depth not in _DEPTH_CONFIGS:
            raise ValueError(
                f"Unsupported depth {depth}. "
                f"Choose one of {sorted(_DEPTH_CONFIGS)}."
            )

        blocks_per_stage = _DEPTH_CONFIGS[depth]
        self.depth = depth

        self.in_channels = 64

        # ----------------------------------------
        # ECG stem
        # ----------------------------------------

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(
                64
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.layer1 = self._make_layer(
            planes=64,
            blocks=blocks_per_stage[0],
            stride=1,
        )

        self.layer2 = self._make_layer(
            planes=128,
            blocks=blocks_per_stage[1],
            stride=2,
        )

        self.layer3 = self._make_layer(
            planes=256,
            blocks=blocks_per_stage[2],
            stride=2,
        )

        self.layer4 = self._make_layer(
            planes=512,
            blocks=blocks_per_stage[3],
            stride=2,
        )

        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        feature_dim = (
            512
            * Bottleneck1D.expansion
        )

        self.features = feature_dim

        # Config specifies dropout=0.20 for the ECG branch, but no dropout
        # was previously wired into the model at all. Applied on the pooled
        # feature vector immediately before the task heads.
        self.dropout = nn.Dropout(p=dropout)

        self.heads = ECGMultiTaskHeads(
            input_dim=feature_dim,
            num_diagnostic_classes=(
                num_diagnostic_classes
            ),
            num_rhythm_classes=(
                num_rhythm_classes
            ),
        )

        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
    ):

        downsample = None

        output_channels = (
            planes
            * Bottleneck1D.expansion
        )

        if (
            stride != 1
            or self.in_channels
            != output_channels
        ):

            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(
                    output_channels
                ),
            )

        layers = [
            Bottleneck1D(
                self.in_channels,
                planes,
                stride,
                downsample,
            )
        ]

        self.in_channels = (
            output_channels
        )

        for _ in range(
            1,
            blocks
        ):

            layers.append(
                Bottleneck1D(
                    self.in_channels,
                    planes,
                )
            )

        return nn.Sequential(
            *layers
        )

    def _initialize_weights(
        self
    ):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv1d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm1d
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(
                module,
                nn.Linear
            ):

                nn.init.normal_(
                    module.weight,
                    0,
                    0.01
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        if x.ndim != 3:
            raise ValueError(
                "Expected ECG input "
                "[batch, channels, length], "
                f"got {tuple(x.shape)}."
            )

        out = self.stem(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)

        out = out.flatten(
            1
        )

        out = self.dropout(out)

        return self.heads(out)