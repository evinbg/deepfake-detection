import torch
import torch.nn as nn
import numpy as np

# Positional Encoding

class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to embeddings."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


# Transformer Encoder Model with Attention Pooling

class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim = 3,
        hidden_dim = 128,
        num_layers = 4,
        num_heads = 4,
        dropout = 0.1
    ):
        super(TransformerEncoderClassifier, self).__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        
        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Linear (128 → 128)
            nn.ReLU(),                          
            nn.Dropout(dropout),                
            nn.Linear(hidden_dim, 2)           # Linear (128 → 2)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # [batch_size, seq_len, hidden_dim]
        
        # Attention pooling
        attn_weights = self.softmax(self.attention(x)) # [batch_size, seq_len, 1]
        x = torch.sum(x * attn_weights, dim=1) # [batch_size, hidden_dim]
        
        # Classification
        x = self.classification_head(x)  # [batch_size, 2]
        return x