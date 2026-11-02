"""
ECG model factories.
"""

from .cnn_bilstm_attention import (
    CNNBiLSTMAttention,
)
from .cnn1d import CNN1D
from .xresnet1d import XResNet1D


def build_cnn_bilstm_attention(
    diagnostic_classes,
    rhythm_classes,
):

    return CNNBiLSTMAttention(

        num_leads=12,

        num_diagnostic_classes=
            len(diagnostic_classes),

        num_rhythm_classes=
            rhythm_classes,

        cnn_channels=128,

        lstm_hidden=128,

        lstm_layers=2,

        dropout=0.30,
    )


def build_cnn1d(diagnostic_classes, rhythm_classes):
    return CNN1D(
        input_channels=12,
        num_diagnostic_classes=len(diagnostic_classes),
        dropout=0.50,
    )


def build_xresnet1d(diagnostic_classes, rhythm_classes):
    return XResNet1D(
        input_channels=12,
        num_diagnostic_classes=len(diagnostic_classes),
        num_rhythm_classes=rhythm_classes,
        depth=34,
        dropout=0.30,
    )