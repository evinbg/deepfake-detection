"""input.py
-----------------------------------------------------------------
Utility for **inference‑time feature extraction**.

This script defines :class:`InputVideoProcessor`, a light‑weight helper that
scans *Capstone/input_videos* for MP4 files, extracts the same motion‑delta &
geometry features used during training, and serialises them to
*input_videos/input_features/motion_features.pkl* so they can be fed straight
into the already‑trained MotionTransformer.

Usage (from project root)::

    python -m input --frames 40 --interval 3

The resulting pickle contains::

    (motion_features: np.ndarray [N, frames-1, 68*2],
     labels:          np.ndarray [N])

where **N** is the number of successfully processed videos. All videos are
assumed to be *real* (label 0); adjust the label logic if you intend to detect
fakes as well.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import dlib
import numpy as np
from mtcnn import MTCNN

# ----------------------------------------------------------------------------
# 0) Constants & directories
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[0]      # Capstone/
INPUT_DIR = ROOT_DIR / "input_videos"
FEATURE_OUT = INPUT_DIR / "input_features"

FRAMES_TO_EXTRACT = 40
INTERVAL = 3

FEATURE_OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1) Initialise detectors once (MTCNN + dlib)
# ----------------------------------------------------------------------------
_mtcnn = MTCNN()
_dlib_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(
    str(ROOT_DIR / "shape_predictor_68_face_landmarks.dat")
)

# ----------------------------------------------------------------------------
# 2) Core helpers (trimmed version of the preprocessing code)
# ----------------------------------------------------------------------------

def _extract_frames(video_path: Path, *, frames: int, interval: int) -> np.ndarray | None:
    """Return RGB frames evenly spaced by *interval*, or *None* if too short."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    needed = frames * interval
    if total < needed:
        print(f"[skip] {video_path.name}: only {total} frames (< {needed}).")
        return None

    out: List[np.ndarray] = []
    for i in range(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(out) if len(out) == frames else None


def _detect_and_crop(face_img: np.ndarray) -> np.ndarray | None:
    """Return a (299,299,3) face crop or *None* if detection fails."""
    results = _mtcnn.detect_faces(face_img)
    if not results:
        return None
    x, y, w, h = results[0]["box"]
    pad_x, pad_y = int(0.2 * w), int(0.2 * h)
    x = max(0, x - pad_x);
    y = max(0, y - pad_y)
    w = min(face_img.shape[1] - x, w + 2 * pad_x)
    h = min(face_img.shape[0] - y, h + 2 * pad_y)
    crop = face_img[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (299, 299))


def _extract_landmarks(face_crop: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    rects = _dlib_detector(gray)
    if len(rects) == 0:
        return None
    shape = _predictor(gray, rects[0])
    return np.array([[p.x, p.y] for p in shape.parts()])


def _motion_and_geom(landmarks_seq: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (motion_vectors, geometric_features)."""
    # geometric distances
    geo = []
    for lm in landmarks_seq:
        eye = np.linalg.norm(lm[36] - lm[45])
        mouth = np.linalg.norm(lm[48] - lm[54])
        chin = np.linalg.norm(lm[30] - lm[8])
        geo.append([eye, mouth, chin])
    geo_arr = np.asarray(geo, dtype=np.float32)

    # motion vectors (delta between consecutive frames)
    deltas = np.diff(np.stack(landmarks_seq, axis=0), axis=0)
    motion = deltas.reshape(deltas.shape[0], -1).astype(np.float32)
    return motion, geo_arr

# ----------------------------------------------------------------------------
# 3) Public class
# ----------------------------------------------------------------------------
class InputVideoProcessor:
    """Process all videos in *input_videos/* and dump a pickle for inference."""

    def __init__(self, frames: int = FRAMES_TO_EXTRACT, interval: int = INTERVAL):
        self.frames = frames
        self.interval = interval
        self.motion: List[np.ndarray] = []
        self.labels: List[int] = []

    # ------------------------------------------------------------------
    def process_all(self) -> Path | None:
        vids = sorted(p for p in INPUT_DIR.glob("*.mp4"))
        if not vids:
            print("No videos found in input_videos/.")
            return None

        for vid in vids:
            self._process_one(vid)

        if not self.motion:
            print("No usable videos – feature file not written.")
            return None

        out_path = FEATURE_OUT / "motion_features.pkl"
        with out_path.open("wb") as f:
            pickle.dump((np.stack(self.motion), np.array(self.labels)), f)
        print(f"Saved features for {len(self.motion)} videos → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    def _process_one(self, vid_path: Path):
        frames = _extract_frames(vid_path, frames=self.frames, interval=self.interval)
        if frames is None:
            return
        lms_seq: List[np.ndarray] = []
        for frm in frames:
            crop = _detect_and_crop(frm)
            if crop is None:
                print(f"[skip] {vid_path.name}: face not found in a frame.")
                return
            lm = _extract_landmarks(crop)
            if lm is None or lm.shape != (68, 2):
                print(f"[skip] {vid_path.name}: landmarks missing.")
                return
            lms_seq.append(lm)
        motion, _ = _motion_and_geom(lms_seq)
        if motion.shape[0] != self.frames - 1:
            print(f"[skip] {vid_path.name}: incomplete landmark sequence.")
            return
        self.motion.append(motion)
        self.labels.append(0)  # real by default
        print(f"[ok]   {vid_path.name}: extracted features.")


# ----------------------------------------------------------------------------
# 4) CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract motion features from videos in input_videos/.")
    parser.add_argument("--frames", type=int, default=FRAMES_TO_EXTRACT)
    parser.add_argument("--interval", type=int, default=INTERVAL)
    args = parser.parse_args()

    processor = InputVideoProcessor(frames=args.frames, interval=args.interval)
    processor.process_all()
