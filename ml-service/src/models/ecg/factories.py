"""
ECG model factories.
"""

from .cnn_bilstm_attention import (
    CNNBiLSTMAttention,
)


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