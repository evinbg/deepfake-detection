"""input.py
-----------------------------------------------------------------
Utility for inference‑time feature extraction.

Adds generation of *geometric_features.pkl* alongside the existing
*motion_features.pkl*.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import dlib
import numpy as np
from mtcnn import MTCNN

# ------------------------------------------------------------------
# Directories & defaults
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent            # Capstone/
INPUT_DIR = ROOT / "input_videos"
OUT_DIR   = INPUT_DIR / "input_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = 40
INTERVAL = 3

# ------------------------------------------------------------------
# Detectors
# ------------------------------------------------------------------
_mtcnn = MTCNN()
_dlib = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(str(ROOT / "shape_predictor_68_face_landmarks.dat"))

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _extract_frames(path: Path, *, n: int, step: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n * step:
        print(f"[skip] {path.name}: only {total} frames (< {n*step}).")
        return None
    out: List[np.ndarray] = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, f = cap.read()
        if not ret:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(out) if len(out) == n else None


def _crop_face(img: np.ndarray) -> np.ndarray | None:
    res = _mtcnn.detect_faces(img)
    if not res:
        return None
    x, y, w, h = res[0]["box"]
    padx, pady = int(0.2*w), int(0.2*h)
    x = max(0, x-padx); y = max(0, y-pady)
    w = min(img.shape[1]-x, w+2*padx)
    h = min(img.shape[0]-y, h+2*pady)
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (299, 299))


def _landmarks(face: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    rects = _dlib(gray)
    if not rects:
        return None
    shape = _predictor(gray, rects[0])
    return np.array([[p.x, p.y] for p in shape.parts()])


def _motion_geom(seq: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    geo = []
    for lm in seq:
        geo.append([
            np.linalg.norm(lm[36]-lm[45]),
            np.linalg.norm(lm[48]-lm[54]),
            np.linalg.norm(lm[30]-lm[8])
        ])
    geo = np.asarray(geo, dtype=np.float32)
    delta = np.diff(np.stack(seq), axis=0).reshape(len(seq)-1, -1).astype(np.float32)
    return delta, geo

# ------------------------------------------------------------------
# Processor class
# ------------------------------------------------------------------
class InputVideoProcessor:
    def __init__(self, frames: int = FRAMES, interval: int = INTERVAL):
        self.frames = frames
        self.interval = interval
        self.motion: List[np.ndarray] = []
        self.geo:    List[np.ndarray] = []
        self.labels: List[int] = []

    def process_all(self) -> Tuple[Path, Path] | None:
        real_vids = sorted((INPUT_DIR / "real").glob("*.mp4"))
        fake_vids = sorted((INPUT_DIR / "fake").glob("*.mp4"))
        vids = real_vids + fake_vids
        if not vids:
            print("No videos in input_videos/.")
            return None
        for v in real_vids:
            self._one(v, label=0)
        for v in fake_vids:
            self._one(v, label=1)
        if not self.motion:
            print("No usable videos.")
            return None
        mot_path = OUT_DIR / "motion_features.pkl"
        geo_path = OUT_DIR / "geometric_features.pkl"
        with mot_path.open("wb") as f:
            pickle.dump((np.stack(self.motion), np.array(self.labels)), f)
        with geo_path.open("wb") as f:
            pickle.dump((np.stack(self.geo), np.array(self.labels)), f)
        print(f"Wrote {len(self.motion)} videos → {mot_path} & {geo_path}")
        return mot_path, geo_path

    def _one(self, path: Path, label: int):
        frames = _extract_frames(path, n=self.frames, step=self.interval)
        if frames is None:
            return
        seq = []
        for f in frames:
            face = _crop_face(f)
            if face is None:
                print(f"[skip] {path.name}: face not found.")
                return
            lm = _landmarks(face)
            if lm is None or lm.shape != (68,2):
                print(f"[skip] {path.name}: landmarks fail.")
                return
            seq.append(lm)
        motion, geo = _motion_geom(seq)
        if motion.shape[0] != self.frames-1:
            print(f"[skip] {path.name}: landmark seq len mismatch.")
            return
        self.motion.append(motion)
        self.geo.append(geo)
        self.labels.append(label)
        print(f"[ok] {path.name}")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--interval", type=int, default=INTERVAL)
    args = p.parse_args()

    proc = InputVideoProcessor(frames=args.frames, interval=args.interval)
    proc.process_all()
