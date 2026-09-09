import torch
import torch.nn as nn


class ECGMultiTaskHeads(
    nn.Module
):
    """
    ECG multitask output heads.

    Diagnostic:
        NORM, MI, STTC, CD, HYP

    Rhythm:
        PTB-XL rhythm labels.
    """

    def __init__(
        self,
        input_dim: int,
        num_diagnostic_classes: int,
        num_rhythm_classes: int,
    ):
        super().__init__()

        if num_diagnostic_classes <= 0:
            raise ValueError(
                "num_diagnostic_classes must be > 0."
            )

        self.diagnostic = nn.Linear(
            input_dim,
            num_diagnostic_classes
        )

        if num_rhythm_classes > 0:

            self.rhythm = nn.Linear(
                input_dim,
                num_rhythm_classes
            )

        else:

            self.rhythm = None

    def forward(
        self,
        x: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        outputs = {
            "diagnostic": self.diagnostic(x)
        }

        if self.rhythm is not None:

            outputs["rhythm"] = (
                self.rhythm(x)
            )

        return outputs