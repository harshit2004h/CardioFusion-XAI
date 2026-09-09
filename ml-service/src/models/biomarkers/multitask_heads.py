import torch
import torch.nn as nn

from ..common.heads import (
    BinaryClassificationHead,
    MulticlassClassificationHead,
)


class BiomarkerMultiTaskHeads(nn.Module):
    """
    Collection of task-specific heads for the
    shared biomarker representation.

    Binary tasks:
        one logit per disease.

    Multiclass tasks:
        one logit per class.
    """

    def __init__(
        self,
        input_dim: int,
        target_specs: dict,
    ):
        super().__init__()

        self.target_specs = target_specs

        self.heads = nn.ModuleDict()

        for target, spec in target_specs.items():

            target_type = spec[
                "type"
            ]

            if target_type == "binary":

                self.heads[target] = (
                    BinaryClassificationHead(
                        input_dim=input_dim
                    )
                )

            elif target_type == "multiclass":

                num_classes = int(
                    spec["num_classes"]
                )

                self.heads[target] = (
                    MulticlassClassificationHead(
                        input_dim=input_dim,
                        num_classes=num_classes,
                    )
                )

            else:

                raise ValueError(
                    f"Unsupported target type "
                    f"'{target_type}' for "
                    f"'{target}'."
                )

    def forward(
        self,
        x: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        return {
            target: head(x)
            for target, head
            in self.heads.items()
        }