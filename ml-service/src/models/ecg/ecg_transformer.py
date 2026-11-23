"""CNN + temporal Transformer encoder for 12-lead PTB-XL ECG."""
from __future__ import annotations

import torch
import torch.nn as nn


class ECGTransformer(nn.Module):
    """Encode local ECG morphology with CNN blocks and long-range context with attention.

    Input shape: ``[batch, 12, 1000]`` for the configured 100 Hz / 10 s input.
    Outputs are raw logits for multi-label BCEWithLogitsLoss.
    """

    def __init__(
        self,
        num_leads: int = 12,
        num_diagnostic_classes: int = 5,
        num_rhythm_classes: int = 0,
        cnn_channels: int = 96,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        max_tokens: int = 250,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.num_leads = num_leads
        self.max_tokens = max_tokens
        self.model_config = {
            "num_leads": num_leads,
            "cnn_channels": cnn_channels,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "max_tokens": max_tokens,
        }
        self.cnn = nn.Sequential(
            nn.Conv1d(num_leads, cnn_channels, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.projection = nn.Linear(cnn_channels, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attention_score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.diagnostic_head = nn.Linear(d_model, num_diagnostic_classes)
        self.rhythm_head = nn.Linear(d_model, num_rhythm_classes) if num_rhythm_classes else None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.num_leads:
            raise ValueError(f"Expected ECG input [B, {self.num_leads}, T], got {tuple(x.shape)}.")
        if not torch.isfinite(x).all():
            raise ValueError("ECG input contains non-finite values.")
        features = self.cnn(x).transpose(1, 2)
        if features.shape[1] > self.max_tokens:
            raise ValueError(
                f"CNN produced {features.shape[1]} tokens, exceeding max_tokens={self.max_tokens}."
            )
        tokens = self.projection(features) + self.position_embedding[:, : features.shape[1], :]
        encoded = self.encoder(tokens)
        weights = torch.softmax(self.attention_score(encoded), dim=1)
        pooled = torch.sum(weights * encoded, dim=1)
        return self.norm(pooled), encoded, weights

    def forward(self, x: torch.Tensor, return_representation: bool = False) -> dict[str, torch.Tensor]:
        representation, encoded, weights = self.encode(x)
        shared = self.shared(representation)
        rhythm_logits = (
            self.rhythm_head(shared)
            if self.rhythm_head is not None
            else shared.new_empty((shared.shape[0], 0))
        )
        outputs = {
            "diagnostic": self.diagnostic_head(shared),
            "rhythm": rhythm_logits,
            "attention": weights,
        }
        if return_representation:
            outputs["representation"] = representation
            outputs["encoded_tokens"] = encoded
        return outputs
