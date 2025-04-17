import torch
import torch.nn as nn
import numpy as np

# Custom Transformer Encoder Layers to Extract Attention Weights

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """
    A slight modification of PyTorch's TransformerEncoderLayer that returns
    the self-attention weights along with the usual output.
    """
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # The default forward pass in nn.TransformerEncoderLayer:
        # self_attn(src, src, src, ...)
        # We'll capture attn_weights by setting need_weights=True.
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,         # <--- we want attention weights
            average_attn_weights=False # <--- do not average across heads
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src, attn_weights

class TransformerEncoderWithAttn(nn.Module):
    """
    Stacks multiple TransformerEncoderLayerWithAttn layers and
    collects their attention weights.
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Returns:
          - output: final layer output of shape (batch_size, seq_len, d_model)
          - attn_maps_list: a list of attention matrices [layer_1, layer_2, ...],
            where each element is shape (batch_size, nhead, seq_len, seq_len).
        """
        attn_maps_list = []
        output = src

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask,
                                       src_key_padding_mask=src_key_padding_mask)
            attn_maps_list.append(attn_weights)

        return output, attn_maps_list


# Positional Encoding

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to embeddings to provide sequence (frame) order information.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer so it’s not trainable but still moves to GPU if needed
        self.register_buffer('pe', pe.unsqueeze(0)) # shape = (1, max_len, d_model)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        We add the encoding up to seq_len to x.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


# Motion Transformer model that returns logits + attention maps

class MotionTransformer(nn.Module):
    """
    Transformer-based model for classification of videos (real vs. fake) 
    using motion-delta features.
    
    Input shape: (batch_size, seq_len=40, feature_dim=136)
    Output: Probability distribution over 2 classes: [Real, Fake].

    We will return both:
      - logits
      - attn_maps_list (list of attention weights for each layer)
    """
    def __init__(
        self,
        feature_dim=136,     # dimension of motion-delta features per frame
        d_model=128,         # internal embedding dimension used by the Transformer
        nhead=2,             # number of attention heads
        num_layers=1,        # number of Transformer encoder layers
        dim_feedforward=256, 
        dropout=0.5,
        num_classes=2
    ):
        super(MotionTransformer, self).__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # 1) Linear projection of input features to d_model
        self.input_linear = nn.Linear(feature_dim, d_model)

        # 2) Positional encoding module
        self.pos_encoder = PositionalEncoding(d_model)

        # 3) Custom Transformer Encoder that returns attention
        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoderWithAttn(encoder_layer, num_layers=num_layers)

        # 4) Classification head: we pool over the sequence dimension (e.g., average pooling)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
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
