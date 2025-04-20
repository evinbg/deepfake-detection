"""
Geometric Transformer for Deepfake Detection
This script implements a Transformer-based model for deepfake detection using geometric features.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

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

# 1. Load and Prepare Data
FEATURES_PATH = 'features/'
# Load geometric features
with open(os.path.join(FEATURES_PATH, 'geometric_features.pkl'), 'rb') as f:
    all_geometric, all_labels = pickle.load(f)

# Debug: Check unique values in all_labels and shape of all_geometric
print(f"Unique labels: {np.unique(all_labels)}")
print(f"Shape of all_geometric: {all_geometric.shape}")
if not np.all(np.isin(all_labels, [0, 1])):
    raise ValueError("Labels must be binary (0 or 1). Found unexpected values.")

# Pad data to ensure 40 frames
def pad_data(data, target_len=40, expected_feature_dim=None):
    padded = []
    for seq in data:
        seq = np.array(seq)
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        elif seq.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got shape {seq.shape}")

        num_frames, feature_dim = seq.shape
        if expected_feature_dim is not None and feature_dim != expected_feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {expected_feature_dim}, got {feature_dim}")

        if num_frames < target_len:
            pad_width = ((0, target_len - num_frames), (0, 0))
            seq = np.pad(seq, pad_width, mode='constant')
        elif num_frames > target_len:
            seq = seq[:target_len]
        padded.append(seq)
    return np.array(padded)

# Pad geometric features to 40 frames
all_geometric = pad_data(all_geometric, target_len=40, expected_feature_dim=3)
assert all_geometric.shape[1] == 40, f"Expected 40 frames, got {all_geometric.shape[1]}"
all_features = all_geometric

# Normalize features
scaler = StandardScaler()
all_features_reshaped = all_features.reshape(-1, all_features.shape[-1])
all_features_scaled = scaler.fit_transform(all_features_reshaped)
all_features = all_features_scaled.reshape(all_features.shape)

# Convert to PyTorch tensors
X = torch.tensor(all_features, dtype=torch.float32)
y = torch.tensor(all_labels, dtype=torch.long)

# Split into train/validation/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42  # 0.1765 of 85% ≈ 15% of total
)

# Custom Dataset
class GeometricDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = GeometricDataset(X_train, y_train)
val_dataset = GeometricDataset(X_val, y_val)
test_dataset = GeometricDataset(X_test, y_test)

# Weighted sampling for training to handle class imbalance
class_counts = np.bincount(y_train)
num_samples = len(y_train)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 2. Define Transformer Encoder Model with Attention Pooling
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerEncoderClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        attn_weights = self.softmax(self.attention(x))
        x = torch.sum(x * attn_weights, dim=1)
        x = self.classification_head(x)
        return x

# Hyperparameters
input_dim = all_features.shape[-1]
hidden_dim = 128
num_layers = 4
num_heads = 4
dropout = 0.2

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoderClassifier(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)

# Compute class weights for loss
total_samples = sum(class_counts)
num_classes = 2
class_weights = torch.tensor([total_samples / (count * num_classes) for count in class_counts], device=device).float()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Learning Rate Scheduler with Warmup
num_epochs = 30
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = num_training_steps // 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


