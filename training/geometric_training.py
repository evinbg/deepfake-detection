"""
Training script for Geometric Transformer for Deepfake Detection
This script loads data, preprocesses it, and trains the Transformer model.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.geometric_transformer import TransformerEncoderClassifier

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and Prepare Data
FEATURES_PATH = 'features/'
def load_and_pad_pkl(pkl_path, target_len=40, expected_feature_dim=3):
    with open(pkl_path, 'rb') as f:
        features, labels = pickle.load(f)
    #features = pad_data(features, target_len=target_len, expected_feature_dim=expected_feature_dim)
    return features, labels

train_features, train_labels = load_and_pad_pkl(os.path.join(FEATURES_PATH, 'train', 'geometric_features.pkl'))
val_features, val_labels = load_and_pad_pkl(os.path.join(FEATURES_PATH, 'val', 'geometric_features.pkl'))
test_features, test_labels = load_and_pad_pkl(os.path.join(FEATURES_PATH, 'test', 'geometric_features.pkl'))

# Normalize separately to avoid data leakage from val/test
scaler = StandardScaler()
train_features_reshaped = train_features.reshape(-1, train_features.shape[-1])
train_features_scaled = scaler.fit_transform(train_features_reshaped)
train_features = train_features_scaled.reshape(train_features.shape)

# Apply same transform to val/test
val_features = scaler.transform(val_features.reshape(-1, val_features.shape[-1])).reshape(val_features.shape)
test_features = scaler.transform(test_features.reshape(-1, test_features.shape[-1])).reshape(test_features.shape)

# Convert to PyTorch tensors
X_train = torch.tensor(train_features, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_val = torch.tensor(val_features, dtype=torch.float32)
y_val = torch.tensor(val_labels, dtype=torch.long)
X_test = torch.tensor(test_features, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

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

# Hyperparameters
input_dim = X_train.shape[-1]
hidden_dim = 192
num_layers = 3
num_heads = 4
dropout = 0.2

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoderClassifier(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)

# Compute class weights for loss
total_samples = sum(class_counts)
num_classes = 2
class_weights = torch.tensor([total_samples / (count * num_classes) for count in class_counts], device=device).float()
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Learning Rate Scheduler with Warmup
num_epochs = 30
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = num_training_steps // 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Evaluation Function
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

# Training Loop
num_epochs = 50
best_val_loss = float('inf')
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
val_precisions, val_recalls, val_f1s, val_roc_aucs = [], [], [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct_preds = 0
    total_preds = 0
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

        # Track training accuracy
        preds = torch.argmax(outputs, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    train_accuracy = 100 * correct_preds / total_preds

    # Validation
    val_metrics = evaluate_model(model, val_loader, device)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_metrics['loss']

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_metrics['accuracy'])
    val_precisions.append(val_metrics['precision'])
    val_recalls.append(val_metrics['recall'])
    val_f1s.append(val_metrics['f1'])
    val_roc_aucs.append(val_metrics['roc_auc'])

    # Save best model
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     torch.save(model.state_dict(), 'outputs/best_transformer_deepfake_model.pth')

    torch.save(model.state_dict(), 'outputs/last_epoch_transformer_model.pth')

    print(f"[Epoch {epoch+1}/{num_epochs}] "
      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
      f"Precision: {val_metrics['precision']:.4f}, "
      f"Recall: {val_metrics['recall']:.4f}, "
      f"F1: {val_metrics['f1']:.4f}, "
      f"ROC-AUC: {val_metrics['roc_auc']:.4f}")

# 5. Plot Metrics Over Epochs
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(epochs, train_accuracies, label='Training Accuracy', color='orange')
plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
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
plt.savefig("outputs/output_metrics_geometric.png", dpi=300, bbox_inches='tight')
plt.close()

# Evaluate Transformer on Validation Set
model.load_state_dict(torch.load('outputs/last_epoch_transformer_model.pth'))
val_metrics = evaluate_model(model, val_loader, device)
test_metrics = evaluate_model(model, test_loader, device)

# Bar Plot for Transformer Validation Metrics Only
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
transformer_scores = [
    val_metrics['accuracy'],
    val_metrics['precision'],
    val_metrics['recall'],
    val_metrics['f1'],
    val_metrics['roc_auc']
]

# Compute confusion matrix
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_test_geometric.png", dpi=300, bbox_inches='tight')
plt.close()

# Print Final Results for Transformer
print("\nTransformer Validation Set Performance:")
print("| Metric     | Value      |")
print("|------------|------------|")
print(f"| Accuracy   | {val_metrics['accuracy']:.2f}%   |")
print(f"| Precision  | {val_metrics['precision']:.4f}   |")
print(f"| Recall     | {val_metrics['recall']:.4f}   |")
print(f"| F1-Score   | {val_metrics['f1']:.4f}   |")
print(f"| ROC-AUC    | {val_metrics['roc_auc']:.4f}   |")
print("")
print("\nPerformance Benchmarking on Test Set:")
print("| Metric     | Value      |")
print("|------------|------------|")
print(f"| Accuracy   | {test_metrics['accuracy']:.2f}%   |")
print(f"| Precision  | {test_metrics['precision']:.4f}   |")
print(f"| Recall     | {test_metrics['recall']:.4f}   |")
print(f"| F1-Score   | {test_metrics['f1']:.4f}   |")
print(f"| ROC-AUC    | {test_metrics['roc_auc']:.4f}   |")