"""motion_transformer.py
-----------------------------------------------------------------
Defines the Transformer‑based model (and supporting layers) used to
classify videos based on motion‑delta features.  Import this module in
motion_training.py to train or evaluate the network.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "TransformerEncoderLayerWithAttn",
    "TransformerEncoderWithAttn",
    "PositionalEncoding",
    "MotionTransformer",
]


# 1) Custom Transformer encoder layer that exposes self‑attention maps
class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """Identical to ``nn.TransformerEncoderLayer`` but also returns the
    per‑head attention weights produced by *self_attn*.
    """

    def forward(  # type: ignore[override]
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- Multi‑head self‑attention -----------------------------------
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep each head separate
        )
        src = self.norm1(src + self.dropout1(attn_out))

        # ---- Feed‑forward -------------------------------------------------
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))

        return src, attn_weights  # shape: (B, n_heads, S, S)


# 2) Encoder wrapper that stacks the above layer and gathers attention
class TransformerEncoderWithAttn(nn.Module):
    """Container for *num_layers* ``TransformerEncoderLayerWithAttn`` that
    returns a list with the attention maps from every layer.
    """

    def __init__(self, encoder_layer: TransformerEncoderLayerWithAttn, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(  # type: ignore[override]
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_maps: list[torch.Tensor] = []
        out = src
        for layer in self.layers:
            out, attn_w = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_maps.append(attn_w)
        return out, attn_maps


# 3) Sinusoidal positional encoding
class PositionalEncoding(nn.Module):
    """Adds sinusoidal position information to a batch of embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S, D)
        return x + self.pe[:, : x.size(1)]


# 4) Main model
class MotionTransformer(nn.Module):
    """Transformer that classifies a sequence of motion‑delta vectors as
    **real (0)** or **fake (1)** and optionally returns all attention maps.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the per‑frame motion‑delta vector (default 136).
    d_model : int
        Embedding size inside the Transformer (default 64).
    nhead : int
        Number of attention heads (default 2).
    num_layers : int
        Number of stacked encoder layers (default 1).
    dim_feedforward : int
        Hidden size of the feed‑forward subnet (default 256).
    dropout : float
        Dropout probability (default 0.5).
    num_classes : int
        Number of output classes (default 2).
    """

    def __init__(
        self,
        feature_dim=136,     # dimension of motion-delta features per frame
        d_model=128,         # internal embedding dimension used by the Transformer
        nhead=8,             # number of attention heads
        num_layers=4,        # number of Transformer encoder layers
        dim_feedforward=256, 
        dropout=0.1,
        num_classes=2
    ):
        super(MotionTransformer, self).__init__()

        self.feature_dim = feature_dim

        # ---- Input projection -------------------------------------------
        self.input_linear = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # ---- Transformer encoder ----------------------------------------
        base_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoderWithAttn(base_layer, num_layers)

        # ---- Classification head ----------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, feature_dim)
        Returns:
          - logits of shape (batch_size, num_classes)
          - attn_maps_list: list of shape (num_layers, batch_size, n_heads, seq_len, seq_len)
        """
        # 1) Project input to d_model
        x = self.input_linear(x) # (batch_size, seq_len, d_model)

        # 2) Add positional encoding
        x = self.pos_encoder(x) # (batch_size, seq_len, d_model)

        # 3) Pass through Transformer encoder (collect attention weights)
        x, attn_maps_list = self.transformer_encoder(x) # (batch_size, seq_len, d_model), [layer_1, ...]

        # 4) Pool over the time dimension (seq_len). Here we do mean pooling.
        x = torch.mean(x, dim=1) # (batch_size, d_model)

        # 5) Classification head
        logits = self.classifier(x) # (batch_size, num_classes)

        return logits, attn_maps_list