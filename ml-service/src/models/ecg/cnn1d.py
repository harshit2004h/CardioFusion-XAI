import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    PTB-XL 1D CNN baseline.

    Architecture adapted from the ECG branch of the
    uploaded PTB-XL 1D CNN Kaggle notebook.

    Input:
        [batch, 12, signal_length]

    Output:
        diagnostic logits for:
        NORM, MI, STTC, CD, HYP
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
        # Kaggle:
        # Conv1D(64, 7)
        # BatchNorm
        # ReLU
        # MaxPool
        #
        # Conv1D(128, 3)
        # BatchNorm
        # ReLU
        # MaxPool
        #
        # Conv1D(256, 3)
        # BatchNorm
        # ReLU
        # --------------------------------------------------

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),
        )

        # Equivalent to Keras GlobalAveragePooling1D
        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        # --------------------------------------------------
        # Kaggle model02 classification head
        #
        # Dense(64)
        # Dense(64)
        # Dropout(0.5)
        # Dense(5, sigmoid)
        #
        # Sigmoid is NOT included here because we use
        # BCEWithLogitsLoss during training.
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                num_diagnostic_classes
            ),
        )

    def forward(self, signal):

        # signal:
        # [batch, 12, time]

        x = self.features(signal)

        # [batch, 256, time]
        x = self.global_pool(x)

        # [batch, 256, 1]
        x = torch.flatten(
            x,
            start_dim=1
        )

        # [batch, 256]
        logits = self.classifier(x)

        return {
            "diagnostic": logits
        }