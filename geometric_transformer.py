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

# 3. Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device).long()
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    avg_val_loss = val_loss / len(dataloader)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'loss': avg_val_loss
    }

# 4. Training Loop with Metrics Tracking
best_val_loss = float('inf')
train_losses, val_losses = [], []
val_accuracies, val_precisions, val_recalls, val_f1s, val_roc_aucs = [], [], [], [], []

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
        scheduler.step()
        train_loss += loss.item()

    # Validation
    val_metrics = evaluate_model(model, val_loader, device)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_metrics['loss']

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_metrics['accuracy'])
    val_precisions.append(val_metrics['precision'])
    val_recalls.append(val_metrics['recall'])
    val_f1s.append(val_metrics['f1'])
    val_roc_aucs.append(val_metrics['roc_auc'])

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_transformer_deepfake_model.pth')

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_metrics['accuracy']:.2f}%, "
          f"Precision: {val_metrics['precision']:.4f}, "
          f"Recall: {val_metrics['recall']:.4f}, "
          f"F1: {val_metrics['f1']:.4f}, "
          f"ROC-AUC: {val_metrics['roc_auc']:.4f}")

# 5. Plot Metrics Over Epochs
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(epochs, val_accuracies, label='Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Over Epochs')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(epochs, val_precisions, label='Precision', color='green')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision Over Epochs')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(epochs, val_recalls, label='Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Validation Recall Over Epochs')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(epochs, val_f1s, label='F1-Score', color='purple')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('Validation F1-Score Over Epochs')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(epochs, val_roc_aucs, label='ROC-AUC', color='orange')
plt.xlabel('Epoch')
plt.ylabel('ROC-AUC')
plt.title('Validation ROC-AUC Over Epochs')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(epochs, train_losses, label='Training Loss', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Baselines for Benchmarking
# Baseline 1: Logistic Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1).numpy()
X_val_flat = X_val.reshape(X_val.shape[0], -1).numpy()
y_train_np = y_train.numpy()
y_val_np = y_val.numpy()

lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train_flat, y_train_np)
lr_preds = lr_model.predict(X_val_flat)
lr_probs = lr_model.predict_proba(X_val_flat)[:, 1]

lr_metrics = {
    'accuracy': 100 * (lr_preds == y_val_np).sum() / len(y_val_np),
    'precision': precision_score(y_val_np, lr_preds),
    'recall': recall_score(y_val_np, lr_preds),
    'f1': f1_score(y_val_np, lr_preds),
    'roc_auc': roc_auc_score(y_val_np, lr_probs)
}

# Baseline 2: LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])
        x = self.fc(x)
        return x

lstm_model = LSTMClassifier(input_dim, hidden_dim, num_layers=2, dropout=0.2).to(device)
optimizer = AdamW(lstm_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Train LSTM
lstm_model.train()
for epoch in range(20):
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = lstm_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

lstm_metrics = evaluate_model(lstm_model, val_loader, device)

# Evaluate Transformer on Validation Set
model.load_state_dict(torch.load('best_transformer_deepfake_model.pth'))
val_metrics = evaluate_model(model, val_loader, device)

# 7. Bar Plot for Model Comparison
models = ['Logistic Regression', 'LSTM', 'Transformer']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
data = {
    'Accuracy': [lr_metrics['accuracy'], lstm_metrics['accuracy'], val_metrics['accuracy']],
    'Precision': [lr_metrics['precision'], lstm_metrics['precision'], val_metrics['precision']],
    'Recall': [lr_metrics['recall'], lstm_metrics['recall'], val_metrics['recall']],
    'F1-Score': [lr_metrics['f1'], lstm_metrics['f1'], val_metrics['f1']],
    'ROC-AUC': [lr_metrics['roc_auc'], lstm_metrics['roc_auc'], val_metrics['roc_auc']]
}

x = np.arange(len(models))
width = 0.15
fig, ax = plt.subplots(figsize=(10, 6))

for i, metric in enumerate(metrics):
    ax.bar(x + i * width, data[metric], width, label=metric)

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison Across Models')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print Final Results
print("\nPerformance Benchmarking on Validation Set:")
print("| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
print("|--------------------|----------|-----------|--------|----------|---------|")
print(f"| Logistic Regression| {lr_metrics['accuracy']:.2f}% | {lr_metrics['precision']:.4f} | {lr_metrics['recall']:.4f} | {lr_metrics['f1']:.4f} | {lr_metrics['roc_auc']:.4f} |")
print(f"| LSTM              | {lstm_metrics['accuracy']:.2f}% | {lstm_metrics['precision']:.4f} | {lstm_metrics['recall']:.4f} | {lstm_metrics['f1']:.4f} | {lstm_metrics['roc_auc']:.4f} |")
print(f"| Transformer       | {val_metrics['accuracy']:.2f}% | {val_metrics['precision']:.4f} | {val_metrics['recall']:.4f} | {val_metrics['f1']:.4f} | {val_metrics['roc_auc']:.4f} |")


















#The old model (bellow)
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Set random seed for reproducibility
# torch.manual_seed(42)

# # Positional Encoding
# class PositionalEncoding(nn.Module):
#     """Adds sinusoidal positional encoding to embeddings."""
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         seq_len = x.size(1)
#         x = x + self.pe[:, :seq_len, :]
#         return x

# # 1. Load and Prepare Data
# FEATURES_PATH = 'features/'
# # Load geometric features
# with open(os.path.join(FEATURES_PATH, 'geometric_features.pkl'), 'rb') as f:
#     all_geometric, all_labels = pickle.load(f)

# # Debug: Check unique values in all_labels and shape of all_geometric
# print(f"Unique labels: {np.unique(all_labels)}")
# print(f"Shape of all_geometric: {all_geometric.shape}")
# if not np.all(np.isin(all_labels, [0, 1])):
#     raise ValueError("Labels must be binary (0 or 1). Found unexpected values.")

# # Pad data to ensure 40 frames
# def pad_data(data, target_len=40, expected_feature_dim=None):
#     padded = []
#     for seq in data:
#         # Ensure seq is a 2D array
#         seq = np.array(seq)
#         if seq.ndim == 1:
#             # If seq is 1D, assume it's (num_frames,) and reshape to (num_frames, 1)
#             seq = seq.reshape(-1, 1)
#         elif seq.ndim != 2:
#             raise ValueError(f"Expected 2D sequence, got shape {seq.shape}")

#         num_frames, feature_dim = seq.shape
#         if expected_feature_dim is not None and feature_dim != expected_feature_dim:
#             raise ValueError(f"Feature dimension mismatch: expected {expected_feature_dim}, got {feature_dim}")

#         # Pad or truncate to target_len frames
#         if num_frames < target_len:
#             pad_width = ((0, target_len - num_frames), (0, 0))
#             seq = np.pad(seq, pad_width, mode='constant')
#         elif num_frames > target_len:
#             seq = seq[:target_len]
#         padded.append(seq)
#     return np.array(padded)

# # Pad geometric features to 40 frames
# all_geometric = pad_data(all_geometric, target_len=40, expected_feature_dim=3)
# assert all_geometric.shape[1] == 40, f"Expected 40 frames, got {all_geometric.shape[1]}"
# all_features = all_geometric  # No motion features, so all_features is just all_geometric

# # Normalize features
# scaler = StandardScaler()
# all_features_reshaped = all_features.reshape(-1, all_features.shape[-1])
# all_features_scaled = scaler.fit_transform(all_features_reshaped)
# all_features = all_features_scaled.reshape(all_features.shape)

# # Convert to PyTorch tensors
# X = torch.tensor(all_features, dtype=torch.float32)  # Shape: [num_videos, num_frames, num_features]
# y = torch.tensor(all_labels, dtype=torch.long)       # Shape: [num_videos]

# # Stratified train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Custom Dataset
# class GeometricDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# train_dataset = GeometricDataset(X_train, y_train)
# test_dataset = GeometricDataset(X_test, y_test)

# # Weighted sampling for training to handle class imbalance
# class_counts = np.bincount(y_train)
# num_samples = len(y_train)
# class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# sample_weights = class_weights[y_train]
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

# train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # 2. Define Transformer Encoder Model with Attention Pooling
# class TransformerEncoderClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):  # Fixed the dropout parameter
#         super(TransformerEncoderClassifier, self).__init__()
        
#         # Input embedding layer
#         self.embedding = nn.Linear(input_dim, hidden_dim)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(hidden_dim)
        
#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Attention pooling
#         self.attention = nn.Linear(hidden_dim, 1)
#         self.softmax = nn.Softmax(dim=1)
        
#         # Classification head sequentially 
#         self.classification_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),  # Linear (128 → 128)
#             nn.ReLU(),                          
#             nn.Dropout(dropout),                
#             nn.Linear(hidden_dim, 2)            # Linear (128 → 2)
#         )
        
#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.pos_encoder(x)
#         x = self.transformer_encoder(x)  # [batch_size, seq_len, hidden_dim]
        
#         # Attention pooling
#         attn_weights = self.softmax(self.attention(x))  # [batch_size, seq_len, 1]
#         x = torch.sum(x * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
#         # Classification
#         x = self.classification_head(x)  # [batch_size, 2]
#         return x

# # Hyperparameters
# input_dim = all_features.shape[-1]  # Should be 3 (geometric features only)
# hidden_dim = 128
# num_layers = 2
# num_heads = 4
# dropout = 0.2

# # Initialize model, loss, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = TransformerEncoderClassifier(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)

# # Compute class weights for loss (for CrossEntropyLoss)
# class_counts = np.bincount(all_labels)
# class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], device=device).float()
# class_weights = class_weights / class_weights.sum() * 2
# print(f"Class weights: {class_weights}")

# # Use CrossEntropyLoss instead of BCELoss
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# # 3. Training Loop with Validation
# num_epochs = 50
# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     train_loss = 0.0
#     for features, labels in train_loader:
#         features, labels = features.to(device), labels.to(device).long()
#         optimizer.zero_grad()
#         outputs = model(features)  # [batch_size, 2]
#         loss = criterion(outputs, labels)  # CrossEntropyLoss expects [batch_size, num_classes] and [batch_size]
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         train_loss += loss.item()

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for features, labels in test_loader:
#             features, labels = features.to(device), labels.to(device).long()
#             outputs = model(features)  # [batch_size, 2]
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             predicted = torch.argmax(outputs, dim=1)  # Get the predicted class by taking argmax
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     avg_train_loss = train_loss / len(train_loader)
#     avg_val_loss = val_loss / len(test_loader)
#     val_accuracy = 100 * correct / total

#     scheduler.step(avg_val_loss)
#     print(f"[Epoch {epoch+1}/{num_epochs}] "
#           f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
#           f"Val Acc: {val_accuracy:.2f}%")

# # Save the model
# torch.save(model.state_dict(), 'transformer_deepfake_model.pth')
# print("Model saved to 'transformer_deepfake_model.pth'")