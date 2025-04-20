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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from models.geometric_transformer import TransformerEncoderClassifier

# Set random seed for reproducibility
torch.manual_seed(42)

# 1. Load and Prepare Data
FEATURES_PATH = 'features/'
with open(os.path.join(FEATURES_PATH, 'geometric_features.pkl'), 'rb') as f:
    all_geometric, all_labels = pickle.load(f)

# Debug: Check data
print(f"Unique labels: {np.unique(all_labels)}")
print(f"Shape of all_geometric: {all_geometric.shape}")
if not np.all(np.isin(all_labels, [0, 1])):
    raise ValueError("Labels must be binary (0 or 1).")

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

all_geometric = pad_data(all_geometric, target_len=40, expected_feature_dim=3)
assert all_geometric.shape[1] == 40, f"Expected 40 frames, got {all_geometric.shape[1]}"
all_features = all_geometric

# Split into train/validation/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_features, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42  # 0.25 of 80% = 20% of total
)

# Normalize features
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Validation class distribution: {np.bincount(y_val)}")
print(f"Testing class distribution: {np.bincount(y_test)}")

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

# Weighted sampling
class_counts = np.bincount(y_train)
num_samples = len(y_train)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 2. Model Setup
input_dim = all_features.shape[-1]  # 3
hidden_dim = 128
num_layers = 2
num_heads = 4
dropout = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoderClassifier(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)

# Loss and optimizer
class_weights_loss = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], device=device).float()
class_weights_loss = class_weights_loss / class_weights_loss.sum() * 2
criterion = nn.CrossEntropyLoss(weight=class_weights_loss)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

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
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
    avg_val_loss = val_loss / len(dataloader)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'loss': avg_val_loss
    }

# 4. Baseline: LSTM
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

# 5. Training Loop
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0
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
    
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_metrics['accuracy']:.2f}%, "
          f"Precision: {val_metrics['precision']:.4f}, "
          f"Recall: {val_metrics['recall']:.4f}, "
          f"F1: {val_metrics['f1']:.4f}, "
          f"ROC-AUC: {val_metrics['roc_auc']:.4f}")
    
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
    
    scheduler.step(avg_val_loss)

# Load best model
model.load_state_dict(torch.load('best_transformer_model.pth'))

# 6. Plot Metrics
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(epochs, val_accuracies, label='Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(epochs, val_precisions, label='Precision', color='green')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(epochs, val_recalls, label='Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Validation Recall')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(epochs, val_f1s, label='F1-Score', color='purple')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('Validation F1-Score')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(epochs, val_roc_aucs, label='ROC-AUC', color='orange')
plt.xlabel('Epoch')
plt.ylabel('ROC-AUC')
plt.title('Validation ROC-AUC')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

# 7. Baselines
# Logistic Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1).numpy()
X_val_flat = X_val.reshape(X_val.shape[0], -1).numpy()
y_train_np = y_train.numpy()
y_val_np = y_val.numpy()

lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train_flat, y_train_np)
lr_preds = lr_model.predict(X_val_flat)
lr_probs = lr_model.predict_proba(X_val_flat)[:, 1]
lr_metrics = {
    'accuracy': accuracy_score(y_val_np, lr_preds) * 100,
    'precision': precision_score(y_val_np, lr_preds, zero_division=0),
    'recall': recall_score(y_val_np, lr_preds, zero_division=0),
    'f1': f1_score(y_val_np, lr_preds, zero_division=0),
    'roc_auc': roc_auc_score(y_val_np, lr_probs) if len(np.unique(y_val_np)) > 1 else 0
}

# LSTM
lstm_model = LSTMClassifier(input_dim, hidden_dim, num_layers=2, dropout=0.2).to(device)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
criterion_lstm = nn.CrossEntropyLoss(weight=class_weights_loss)

for epoch in range(20):
    lstm_model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).long()
        optimizer_lstm.zero_grad()
        outputs = lstm_model(features)
        loss = criterion_lstm(outputs, labels)
        loss.backward()
        optimizer_lstm.step()

lstm_metrics = evaluate_model(lstm_model, val_loader, device)

# Transformer Metrics
transformer_metrics = evaluate_model(model, val_loader, device)

# 8. Plot Model Comparison
models = ['Logistic Regression', 'LSTM', 'Transformer']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
data = {
    'Accuracy': [lr_metrics['accuracy'], lstm_metrics['accuracy'], transformer_metrics['accuracy']],
    'Precision': [lr_metrics['precision'], lstm_metrics['precision'], transformer_metrics['precision']],
    'Recall': [lr_metrics['recall'], lstm_metrics['recall'], transformer_metrics['recall']],
    'F1-Score': [lr_metrics['f1'], lstm_metrics['f1'], transformer_metrics['f1']],
    'ROC-AUC': [lr_metrics['roc_auc'], lstm_metrics['roc_auc'], transformer_metrics['roc_auc']]
}

x = np.arange(len(models))
width = 0.15
fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, data[metric], width, label=metric)
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Print Final Results
print("\nValidation Set Performance:")
print("| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
print("|--------------------|----------|-----------|--------|----------|---------|")
print(f"| Logistic Regression| {lr_metrics['accuracy']:.2f}% | {lr_metrics['precision']:.4f} | {lr_metrics['recall']:.4f} | {lr_metrics['f1']:.4f} | {lr_metrics['roc_auc']:.4f} |")
print(f"| LSTM              | {lstm_metrics['accuracy']:.2f}% | {lstm_metrics['precision']:.4f} | {lstm_metrics['recall']:.4f} | {lstm_metrics['f1']:.4f} | {lstm_metrics['roc_auc']:.4f} |")
print(f"| Transformer       | {transformer_metrics['accuracy']:.2f}% | {transformer_metrics['precision']:.4f} | {transformer_metrics['recall']:.4f} | {transformer_metrics['f1']:.4f} | {transformer_metrics['roc_auc']:.4f} |")
