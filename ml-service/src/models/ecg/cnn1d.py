import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    PTB-XL 1D CNN baseline.

    Architecture adapted from the ECG branch of the
    uploaded PTB-XL 1D CNN notebook.

    Input:
        [batch, 12, signal_length]

    Output:
        dictionary containing diagnostic logits
        for the five PTB-XL diagnostic superclasses.
    """

    def __init__(
        self,
        input_channels=12,
        num_diagnostic_classes=5,
        dropout=0.5,
    ):
        super().__init__()

        # --------------------------------------------------
        # ECG feature extractor
        #
        # Corresponds to the Kaggle create_Y_model():
        #
        # Conv1D(64, 7)
        # BN -> ReLU -> MaxPool
        #
        # Conv1D(128, 3)
        # BN -> ReLU -> MaxPool
        #
        # Conv1D(256, 3)
        # BN -> ReLU
        # --------------------------------------------------

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),
        )

        # Equivalent to Keras GlobalAveragePooling1D
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --------------------------------------------------
        # Classification head
        #
        # Adapted from the Kaggle model02 head after
        # removing the metadata branch.
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                64,
                num_diagnostic_classes,
            ),
        )

    def forward(self, signal):

        # signal:
        # [batch, channels, time]

        x = self.features(signal)

        # [batch, 256, time]
        x = self.global_pool(x)

        # [batch, 256, 1]
        x = torch.flatten(
            x,
            start_dim=1,
        )

        # [batch, 256]
        diagnostic_logits = self.classifier(x)

        return {
            "diagnostic": diagnostic_logits,
        }