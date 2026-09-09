import torch
import torch.nn as nn

from .input_adapters import (
    SourceInputAdapter
)

from .multitask_heads import (
    BiomarkerMultiTaskHeads
)


class BiomarkerFTTransformer(
    nn.Module
):
    """
    Shared FT-Transformer-style model
    for the four tabular datasets:

        Zheen
        UCI Heart Failure
        MI Complications
        Framingham

    Each source has its own input adapter.
    The transformer trunk and task heads are shared.
    """

    def __init__(
        self,
        input_dimensions: dict[str, int],
        target_specs: dict,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        if not input_dimensions:
            raise ValueError(
                "input_dimensions cannot be empty."
            )

        self.source_names = list(
            input_dimensions.keys()
        )

        self.d_token = d_token

        # ----------------------------------------
        # Source-specific feature adapters
        # ----------------------------------------

        self.adapters = nn.ModuleDict({
            source: SourceInputAdapter(
                num_features=num_features,
                d_token=d_token,
            )
            for source, num_features
            in input_dimensions.items()
        })

        # ----------------------------------------
        # Shared Transformer encoder
        # ----------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=d_token * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
            )
        )

        self.norm = nn.LayerNorm(
            d_token
        )

        # ----------------------------------------
        # Shared task heads
        # ----------------------------------------

        self.heads = (
            BiomarkerMultiTaskHeads(
                input_dim=d_token,
                target_specs=target_specs,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        source: str,
    ) -> dict[str, torch.Tensor]:

        if source not in self.adapters:
            raise KeyError(
                f"Unknown source '{source}'. "
                f"Available sources: "
                f"{self.source_names}"
            )

        tokens = self.adapters[
            source
        ](x)

        encoded = self.transformer(
            tokens
        )

        # Mean pooling across feature tokens.
        representation = (
            encoded.mean(dim=1)
        )

        representation = self.norm(
            representation
        )

        outputs = self.heads(
            representation
        )

        return outputs