"""Training utilities for CardioFusion-XAI.

The training layer is deliberately modality-specific:
- biomarkers: masked multi-task learning across four disjoint tabular schemas
- ECG: multi-label PTB-XL classification
- echo: LVEF regression on EchoNet-Dynamic

All validation/test evaluation is kept separate from training-time augmentation
and oversampling to reduce leakage.
"""
__all__ = [
    "Trainer",
    "EarlyStopping",
    "MaskedBCEWithLogitsLoss",
    "MaskedCrossEntropyLoss",
    "FocalLoss",
    "MultiTaskLoss",
]


def __getattr__(name: str):
    """Load PyTorch-dependent training components only when they are requested.

    Dataset schema and split utilities remain usable in lightweight environments
    that intentionally do not install PyTorch.
    """
    if name in {"Trainer", "EarlyStopping"}:
        from .trainer import EarlyStopping, Trainer

        return {"Trainer": Trainer, "EarlyStopping": EarlyStopping}[name]
    if name in {"MaskedBCEWithLogitsLoss", "MaskedCrossEntropyLoss", "FocalLoss", "MultiTaskLoss"}:
        from .losses import FocalLoss, MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MultiTaskLoss

        return {
            "MaskedBCEWithLogitsLoss": MaskedBCEWithLogitsLoss,
            "MaskedCrossEntropyLoss": MaskedCrossEntropyLoss,
            "FocalLoss": FocalLoss,
            "MultiTaskLoss": MultiTaskLoss,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
