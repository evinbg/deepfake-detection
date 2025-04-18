"""
Training script for Geometric Transformer for Deepfake Detection
This script loads data, preprocesses it, and trains the Transformer model.
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

# Import the Transformer model
from geometric_transformer import TransformerEncoderClassifier

# Set random seed for reproducibility
torch.manual_seed(42)

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

# Convert to PyTorch tensors
X = torch.tensor(all_features, dtype=torch.float32)  # Shape: [num_videos, num_frames, num_features]
y = torch.tensor(all_labels, dtype=torch.long)       # Shape: [num_videos]

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")

# Normalize features (fit on training data only to avoid data leakage)
scaler = StandardScaler()
# Reshape to 2D for StandardScaler 
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
scaler.fit(X_train_reshaped)
# Transform both training and testing data
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Convert back to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

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
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Initialize early stopping variables
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Training Loop with Validation
num_epochs = 50
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    correct_per_class = [0, 0]
    total_per_class = [0, 0]
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device).long()
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                total_per_class[labels[i].item()] += 1
                if predicted[i] == labels[i]:
                    correct_per_class[labels[i].item()] += 1

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct / total
    class_0_accuracy = 100 * correct_per_class[0] / total_per_class[0] if total_per_class[0] > 0 else 0
    class_1_accuracy = 100 * correct_per_class[1] / total_per_class[1] if total_per_class[1] > 0 else 0

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")
    print(f"Class 0 Acc: {class_0_accuracy:.2f}%, Class 1 Acc: {class_1_accuracy:.2f}%")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_transformer_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Load the best model
model.load_state_dict(torch.load('best_transformer_model.pth'))
print("Best model loaded from 'best_transformer_model.pth'")
# Save the final model
torch.save(model.state_dict(), 'transformer_deepfake_model.pth')
print("Final model saved to 'transformer_deepfake_model.pth'")