"""motion_training.py
-----------------------------------------------------------------
Handles data loading, training, evaluation, and visualization for the
MotionTransformer defined in *motion_transformer.py*.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchviz import make_dot

# Import the model implementation that lives in the sibling file
from motion_transformer import MotionTransformer

__all__ = [
    "MotionDataset",
    "train_transformer_model",
    "visualize_architecture",
    "visualize_attention",
]


#######################################################################
# 1) Dataset wrapper
#######################################################################

class MotionDataset(Dataset):
    """Thin ``torch.utils.data.Dataset`` around a tensor / ndarray pair."""

    def __init__(self, motion: np.ndarray, labels: Sequence[int | np.integer]):
        if motion.ndim != 3:
            raise ValueError("motion must be (N, seq_len, feature_dim)")
        if len(motion) != len(labels):
            raise ValueError("motion and labels length mismatch")
        self.motion = motion.astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.motion)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.from_numpy(self.motion[idx])  # (S, F) → float32
        y = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return x, y


#######################################################################
# 2) Training routine
#######################################################################

def train_transformer_model(
    motion_features_path: str | Path,
    *,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MotionTransformer:
    """Convenience function that trains ``MotionTransformer`` on the feature
    file saved by the preprocessing pipeline.  The pickle must contain a
    tuple ``(all_motion, all_labels)`` where:

    * ``all_motion`` → ndarray (N, 40, 136)
    * ``all_labels`` → ndarray (N,)
    """

    # ---- 1) Load data --------------------------------------------------
    motion_path = Path(motion_features_path)
    if not motion_path.exists():
        raise FileNotFoundError(motion_path)
    with motion_path.open("rb") as f:
        all_motion, all_labels = pickle.load(f)

    num_samples = len(all_motion)
    split_idx = int(0.8 * num_samples)
    train_data = MotionDataset(all_motion[:split_idx], all_labels[:split_idx])
    val_data = MotionDataset(all_motion[split_idx:], all_labels[split_idx:])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # ---- 2) Initialize model / optim / loss ---------------------------
    model = MotionTransformer(feature_dim=all_motion.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_losses, val_losses, val_accs = [], [], []

    # ---- 3) Epoch loop -------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ---- Validation ---------------------------------------------
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100.0 * correct / len(val_data))

        print(
            f"[Epoch {epoch:3d}/{num_epochs}] "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f} | "
            f"Val Acc:  {val_accs[-1]:.2f}%",
        )

    # ---- 4) Plot curves -----------------------------------------------
    _plot_training_curves(train_losses, val_losses, val_accs)
    print("Training complete.")
    return model


def _plot_training_curves(train_l, val_l, val_a):
    epochs = range(1, len(train_l) + 1)
    plt.figure()
    plt.plot(epochs, train_l, label="Train Loss")
    plt.plot(epochs, val_l, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.figure()
    plt.plot(epochs, val_a, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.show()


#######################################################################
# 3) Visualisation helpers
#######################################################################

def visualize_architecture(model: MotionTransformer, save_path: str = "motion_transformer_graph") -> None:
    """Save a *torchviz* graph of the model."""
    dummy = torch.randn(1, 39, model.feature_dim)
    logits, _ = model(dummy)
    graph = make_dot(logits, params=dict(model.named_parameters()))
    graph.render(save_path, format="png")
    print(f"Architecture graph saved to {save_path}.png")


def visualize_attention(model: MotionTransformer, device: str = "cpu", num_frames: int = 39) -> None:
    model.eval()
    dummy = torch.randn(1, num_frames, model.feature_dim).to(device)
    with torch.no_grad():
        _, attn_maps = model(dummy)

    for layer_idx, attn in enumerate(attn_maps):
        n_heads = attn.shape[1]
        for head_idx in range(n_heads):
            plt.figure()
            plt.imshow(attn[0, head_idx].cpu(), aspect="auto")
            plt.colorbar()
            plt.title(f"Layer {layer_idx + 1}, Head {head_idx + 1} Attention")
            plt.xlabel("Key Frames")
            plt.ylabel("Query Frames")
            plt.show()


#######################################################################
# 4) Entry point
#######################################################################

if __name__ == "__main__":
    trained = train_transformer_model(
        motion_features_path="features/motion_features.pkl",
        batch_size=32,
        num_epochs=100,
        learning_rate=1e-4,
    )

    # Print a layer‑by‑layer summary
    summary(trained, input_size=(10, 39, 136))

    # Example visualisations (comment out if not required)
    # visualize_architecture(trained)
    # visualize_attention(trained, device="cuda" if torch.cuda.is_available() else "cpu")
