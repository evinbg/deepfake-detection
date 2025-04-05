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
        # Ensure seq is a 2D array
        seq = np.array(seq)
        if seq.ndim == 1:
            # If seq is 1D, assume it's (num_frames,) and reshape to (num_frames, 1)
            seq = seq.reshape(-1, 1)
        elif seq.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got shape {seq.shape}")

        num_frames, feature_dim = seq.shape
        if expected_feature_dim is not None and feature_dim != expected_feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {expected_feature_dim}, got {feature_dim}")

        # Pad or truncate to target_len frames
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
all_features = all_geometric  # No motion features, so all_features is just all_geometric

# Normalize features
scaler = StandardScaler()
all_features_reshaped = all_features.reshape(-1, all_features.shape[-1])
all_features_scaled = scaler.fit_transform(all_features_reshaped)
all_features = all_features_scaled.reshape(all_features.shape)

# Convert to PyTorch tensors
X = torch.tensor(all_features, dtype=torch.float32)  # Shape: [num_videos, num_frames, num_features]
y = torch.tensor(all_labels, dtype=torch.long)       # Shape: [num_videos]

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
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
test_dataset = GeometricDataset(X_test, y_test)

# Weighted sampling for training to handle class imbalance
class_counts = np.bincount(y_train)
num_samples = len(y_train)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 2. Define Transformer Encoder Model with Attention Pooling
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):  # Fixed the dropout parameter
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
        
        # Attention pooling
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        
        # Classification head sequentially 
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Linear (128 → 128)
            nn.ReLU(),                          
            nn.Dropout(dropout),                
            nn.Linear(hidden_dim, 2)            # Linear (128 → 2)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [batch_size, seq_len, hidden_dim]
        
        # Attention pooling
        attn_weights = self.softmax(self.attention(x))  # [batch_size, seq_len, 1]
        x = torch.sum(x * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        x = self.classification_head(x)  # [batch_size, 2]
        return x

# Hyperparameters
input_dim = all_features.shape[-1]  # Should be 3 (geometric features only)
hidden_dim = 128
num_layers = 2
num_heads = 4
dropout = 0.2

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoderClassifier(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)

# Compute class weights for loss (for CrossEntropyLoss)
class_counts = np.bincount(all_labels)
class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], device=device).float()
class_weights = class_weights / class_weights.sum() * 2
print(f"Class weights: {class_weights}")

# Use CrossEntropyLoss instead of BCELoss
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# 3. Training Loop with Validation
num_epochs = 50
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(features)  # [batch_size, 2]
        loss = criterion(outputs, labels)  # CrossEntropyLoss expects [batch_size, num_classes] and [batch_size]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device).long()
            outputs = model(features)  # [batch_size, 2]
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)  # Get the predicted class by taking argmax
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct / total

    scheduler.step(avg_val_loss)
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), 'transformer_deepfake_model.pth')
print("Model saved to 'transformer_deepfake_model.pth'")