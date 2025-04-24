"""main.py
-----------------------------------------------------------------
Entry point for running inference with a trained MotionTransformer.

* Loads a *.pth* checkpoint from *saved_models/*.
* Runs :class:`InputVideoProcessor` to obtain motion‑delta features for all
  videos in *input_videos/*.
* Prints a per‑video prediction ("Real" or "Fake" + confidence).

Run from the project root::

    python -m main \
        --checkpoint saved_models/motion_transformer_20250424_163432.pth \
        --frames 40 --interval 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 0) Local imports (without breaking when run as module)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[0]     # Capstone/
if str(ROOT / "models") not in sys.path:
    sys.path.append(str(ROOT / "models"))

from models.motion_transformer import MotionTransformer  # type: ignore
from input import InputVideoProcessor  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------
# 1) Helper to map logits → human‑readable
# ---------------------------------------------------------------------
LABELS = {0: "Real", 1: "Fake"}


def _predict(model: MotionTransformer, feats: np.ndarray, device: str) -> List[str]:
    """Return list of label strings for each set of motion features."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats).to(device)          # (N, 39, 136)
        logits, _ = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # (N, 2)
    out = []
    for p in probs:
        cls = int(p.argmax())
        conf = p[cls] * 100
        out.append(f"{LABELS[cls]} ({conf:.1f}%)")
    return out


# ---------------------------------------------------------------------
# 2) CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MotionTransformer on videos in input_videos/.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to .pth file under saved_models/.")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--interval", type=int, default=3)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------
    # 2a) Extract features
    # -----------------------------------------------------------------
    processor = InputVideoProcessor(frames=args.frames, interval=args.interval)
    feat_path = processor.process_all()
    if feat_path is None:
        sys.exit(0)

    motion_feats, _ = np.load(feat_path, allow_pickle=True)
    if motion_feats.ndim != 3:
        print("Unexpected feature shape", motion_feats.shape)
        sys.exit(1)

    # -----------------------------------------------------------------
    # 2b) Load model & weights
    # -----------------------------------------------------------------
    feature_dim = motion_feats.shape[2]
    model = MotionTransformer(feature_dim=feature_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint {args.checkpoint}.")

    # -----------------------------------------------------------------
    # 2c) Predict and display
    # -----------------------------------------------------------------
    results = _predict(model, motion_feats, device)
    vids = sorted(p.name for p in (ROOT / "input_videos").glob("*.mp4"))

    print("\nResults:")
    for name, res in zip(vids, results):
        print(f"  {name:<40} →  {res}")
