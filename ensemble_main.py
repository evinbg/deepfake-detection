"""
ensemble_main.py — 3‑way vote (0.8 Spatial, 0.1 Motion, 0.1 Geo)
----------------------------------------------------------------
Loads the fine‑tuned Spatial/Xception, Motion Transformer, **and**
Geometric Transformer check‑points and computes

    FAKE_prob = 0.80·Spatial + 0.10·Motion + 0.10·Geo

If FAKE_prob ≥ 0.5 → label = "Fake".

Usage
-----
```bash
python -m ensemble_main \
  --motion_ckpt   motion_transformer_20250424_163432.pth \
  --geo_ckpt      last_epoch_transformer_model.pth \
  --spatial_ckpt  fine_tuned_xception.h5
```
(Place the two *.pth* and *.h5* files in the repo root or provide full paths.)
"""
from __future__ import annotations

import argparse
import pickle
import sys
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import importlib  # lazy TF import
from tensorflow.keras.applications.xception import preprocess_input

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT / "models") not in sys.path:
    sys.path.append(str(ROOT / "models"))

from models.motion_transformer import MotionTransformer
from models.geometric_transformer import TransformerEncoderClassifier
from models.spatial_model import SpatialXceptionFT
from input import InputVideoProcessor

LABELS = {0: "Real", 1: "Fake"}
MOTION_W  = 0.10
GEO_W     = 0.10
SPATIAL_W = 0.80
assert abs(MOTION_W + GEO_W + SPATIAL_W - 1.0) < 1e-6

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _predict_motion(model: MotionTransformer, feats: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats).to(device)
        logits, _ = model(x)
        return F.softmax(logits, dim=1).cpu().numpy()[:, 1]


def _predict_geo(model: TransformerEncoderClassifier, feats: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats).to(device)
        logits = model(x)
        return F.softmax(logits, dim=1).cpu().numpy()[:, 1]


def _predict_spatial(model, video_frame_seqs: List[np.ndarray]) -> np.ndarray:
    results = []
    for frame_seq in video_frame_seqs:
        probs = model.predict(frame_seq, verbose=0)
        if probs.ndim == 2 and probs.shape[1] == 2:
            fake_probs = probs[:, 1]
        else:
            fake_probs = probs.squeeze()
        results.append(np.mean(fake_probs))
    return np.array(results)


def _sample_spatial_frames(video_paths: List[Path], frames: int, interval: int) -> List[np.ndarray]:
    import cv2
    all_video_frames = []
    for path in video_paths:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < frames * interval:
            raise RuntimeError(f"{path.name} has only {total} frames (< {frames*interval}).")
        seq = []
        for i in range(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ok, f = cap.read()
            if not ok:
                raise RuntimeError(f"Failed reading {path.name} @ frame {i*interval}.")
            f = cv2.resize(f, (int(f.shape[1]), int(f.shape[0]))) # keep original resolution first
            h, w = f.shape[:2]

            # 20% zoom then crop 80% center region
            zoom_factor = 0.8
            ch, cw = int(h * zoom_factor), int(w * zoom_factor)
            start_y, start_x = (h - ch) // 2, (w - cw) // 2
            cropped = f[start_y:start_y+ch, start_x:start_x+cw]

            # Resize the zoomed-in crop to 299x299 and normalize
            f = cv2.cvtColor(cv2.resize(cropped, (299, 299)), cv2.COLOR_BGR2RGB)
            f = preprocess_input(f.astype(np.float32)) # preprocess for Xception
            seq.append(f)
        all_video_frames.append(np.stack(seq)) # shape: [frames, 299, 299, 3]
    return all_video_frames


def _load_feats(mot_pkl: Path, geo_pkl: Path) -> Tuple[np.ndarray, np.ndarray]:
    with mot_pkl.open("rb") as f:
        mot, _ = pickle.load(f)
    with geo_pkl.open("rb") as f:
        geo, _ = pickle.load(f)
    return mot.astype(np.float32), geo.astype(np.float32)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_ckpt", type=Path, default="saved_models/motion_best.pth")
    ap.add_argument("--geo_ckpt", type=Path, default="saved_models/geometric_best.pth")
    ap.add_argument("--spatial_ckpt", type=Path, default="saved_models/fine_tuned_xception.h5")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--interval", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) features
    proc = InputVideoProcessor(frames=args.frames, interval=args.interval)
    mot_pkl, geo_pkl = proc.process_all()
    mot_feats, geo_feats = _load_feats(mot_pkl, geo_pkl)

    # 2) motion model
    motion_model = MotionTransformer(feature_dim=mot_feats.shape[2]).to(device)
    motion_model.load_state_dict(torch.load(args.motion_ckpt, map_location=device))

    # 3) geometric model (hyper‑params must match training)
    geo_model = TransformerEncoderClassifier(
        input_dim=geo_feats.shape[2],
        hidden_dim=192, # matches geometric_best.pth
        num_layers=4,
        num_heads=8
    ).to(device)

    geo_state = torch.load(args.geo_ckpt, map_location=device)
    geo_model.load_state_dict(geo_state, strict=False) # ignore any extra keys


    # 4) spatial model
    tf = importlib.import_module("tensorflow")
    spatial_model = SpatialXceptionFT().build_model()
    spatial_model.load_weights(str(args.spatial_ckpt))

    # 5) spatial inputs
    real_folder = ROOT / "input_videos" / "real"
    fake_folder = ROOT / "input_videos" / "fake"

    real_vids = sorted(real_folder.glob("*.mp4"))
    real_labels = [0] * len(real_vids)

    fake_vids = sorted(fake_folder.glob("*.mp4"))
    fake_labels = [1] * len(fake_vids)

    vids = real_vids + fake_vids
    gt_labels = real_labels + fake_labels

    spatial_inputs = _sample_spatial_frames(vids, args.frames, args.interval)

    # 6) predictions
    p_motion  = _predict_motion(motion_model, mot_feats, device)
    p_geo     = _predict_geo(geo_model, geo_feats, device)
    p_spatial = _predict_spatial(spatial_model, spatial_inputs)

    fake_prob = (MOTION_W  * p_motion +
                 GEO_W     * p_geo +
                 SPATIAL_W * p_spatial)
    labels = (fake_prob >= 0.5).astype(int)
    conf   = np.where(labels == 1, fake_prob, 1 - fake_prob) * 100

    print("\nWeighted vote (0.8 Spatial / 0.1 Motion / 0.1 Geo):")
    for name, pred, conf_val, gt in zip([v.name for v in vids], labels, conf, gt_labels):
        pred_str = LABELS[int(pred)]
        gt_str = LABELS[int(gt)]
        print(f"  {name:<20} →  {pred_str} ({conf_val:.1f}% conf) [GT: {gt_str}]")

    # Save results to CSV
    csv_path = ROOT / "results.csv"
    with csv_path.open("w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Prediction", "Confidence (%)", "Ground Truth"])
        for v, pred, conf_val, gt in zip(vids, labels, conf, gt_labels):
            writer.writerow([v.name, LABELS[int(pred)], f"{conf_val:.1f}", LABELS[int(gt)]])
    print(f"\n[INFO] Results saved to {csv_path}")

if __name__ == "__main__":
    main()
