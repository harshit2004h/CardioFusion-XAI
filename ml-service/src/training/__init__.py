"""Training utilities for CardioFusion-XAI.

The training layer is deliberately modality-specific:
- biomarkers: masked multi-task learning across four disjoint tabular schemas
- ECG: multi-label PTB-XL classification
- echo: LVEF regression on EchoNet-Dynamic

All validation/test evaluation is kept separate from training-time augmentation
and oversampling to reduce leakage.
"""
from .trainer import Trainer, EarlyStopping
from .losses import (
    MaskedBCEWithLogitsLoss,
    MaskedCrossEntropyLoss,
    FocalLoss,
    MultiTaskLoss,
)

__all__ = [
    "Trainer",
    "EarlyStopping",
    "MaskedBCEWithLogitsLoss",
    "MaskedCrossEntropyLoss",
    "FocalLoss",
    "MultiTaskLoss",
]
