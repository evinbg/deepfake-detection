"""motion_training.py
-----------------------------------------------------------------
Handles data loading, training, evaluation, visualisation, and
persistence for the MotionTransformer defined in *motion_transformer.py*.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Sequence
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------
# 0) Import the model (relative import keeps the package structure clear)
# ---------------------------------------------------------------------
try:
    from ..models.motion_transformer import MotionTransformer   # when run as a module
except ImportError:                                             # fallback for direct call
    sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
    from motion_transformer import MotionTransformer            # type: ignore

__all__ = [
    "MotionDataset",
    "train_transformer_model",
    "visualize_architecture",
    "visualize_attention",
]

# ---------------------------------------------------------------------
# 1) Dataset wrapper
# ---------------------------------------------------------------------
class MotionDataset(Dataset):
    """Thin ``torch.utils.data.Dataset`` around a tensor / ndarray pair."""

    def __init__(self, motion: np.ndarray, labels: Sequence[int | np.integer]):
        if motion.ndim != 3:
            raise ValueError("motion must be (N, seq_len, feature_dim)")
        if len(motion) != len(labels):
            raise ValueError("motion and labels length mismatch")
        self.motion = motion.astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.motion)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.from_numpy(self.motion[idx])  # (S, F) → float32
        y = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------
# 2) Utility: model saver
# ---------------------------------------------------------------------

def _save_model(model: nn.Module, out_dir: Path, prefix: str = "motion_transformer") -> Path:
    """Save *state_dict* to *out_dir/prefix_YYYYmmdd_HHMMSS.pth* and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = out_dir / f"{prefix}_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------
# 3) Training routine
# ---------------------------------------------------------------------

def train_transformer_model(
    motion_features_path: Path,
    *,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: Path | None = None,
) -> MotionTransformer:
    """Train ``MotionTransformer`` on a pickled feature file and optionally save it."""
    if not motion_features_path.exists():
        raise FileNotFoundError(motion_features_path)

    # -----------------------------------------------------------------
    # 1) Load data
    # -----------------------------------------------------------------
    with motion_features_path.open("rb") as f:
        all_motion, all_labels = pickle.load(f)

    split_idx = int(0.8 * len(all_motion))
    train_loader = DataLoader(
        MotionDataset(all_motion[:split_idx], all_labels[:split_idx]),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        MotionDataset(all_motion[split_idx:], all_labels[split_idx:]),
        batch_size=batch_size, shuffle=False
    )

    # -----------------------------------------------------------------
    # 2) Model / loss / optimiser
    # -----------------------------------------------------------------
    model = MotionTransformer(feature_dim=all_motion.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_losses, val_losses, val_accs = [], [], []

    # -----------------------------------------------------------------
    # 3) Epoch loop
    # -----------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # -------- Training ----------
        model.train(); running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            running_loss += loss.item()

        # -------- Validation --------
        model.eval(); val_loss = correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                val_loss += criterion(logits, y).item()
                correct += (logits.argmax(1) == y).sum().item()

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * correct / len(val_loader.dataset))

        print(f"[{epoch:3d}/{num_epochs}] "
              f"Train {train_losses[-1]:.4f} | "
              f"Val {val_losses[-1]:.4f} | "
              f"Acc {val_accs[-1]:.2f}%")

    # -----------------------------------------------------------------
    # 4) Curves & saving
    # -----------------------------------------------------------------
    _plot_training_curves(train_losses, val_losses, val_accs)

    if save_dir is not None:
        _save_model(model, save_dir)

    return model


def _plot_training_curves(train_l, val_l, val_a):
    epochs = range(1, len(train_l) + 1)
    plt.figure(); plt.plot(epochs, train_l, label="Train"); plt.plot(epochs, val_l, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

    plt.figure(); plt.plot(epochs, val_a)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation Accuracy")
    plt.show()


# ---------------------------------------------------------------------
# 4) Entry point with argparse
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]          # Capstone/
    DEFAULT_PKL = ROOT / "features" / "motion_features.pkl"
    DEFAULT_SAVE = ROOT / "saved_models"

    parser = argparse.ArgumentParser(
        description="Train MotionTransformer on motion-delta features and save the model.")
    parser.add_argument("--features", type=Path, default=DEFAULT_PKL,
                        help=f"Path to pickled features (default: {DEFAULT_PKL})")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE,
                        help=f"Directory to place *.pth files (default: {DEFAULT_SAVE})")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()

    model = train_transformer_model(
        motion_features_path=args.features,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
    )

    # Optional: torchinfo summary if installed
    try:
        from torchinfo import summary
        summary(model, input_size=(10, 39, 136))
    except ImportError:
        pass
