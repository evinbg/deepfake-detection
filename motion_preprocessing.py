# motion_preprocessing.py

import os
import cv2
import numpy as np
from mtcnn import MTCNN  # MTCNN for face detection
import dlib             # dlib for facial landmark detection
import pickle
import random

# Global variables
frames_to_extract = 40
num_videos = 590

# Paths
DATASET_PATH = 'datasets/celeb'
FEATURES_PATH = 'features/'
os.makedirs(FEATURES_PATH, exist_ok=True)

# Initialize MTCNN for face detection
detector = MTCNN()

# Load dlib's face detector and pre-trained facial landmark predictor
detector_dlib = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)


def extract_frames(video_path, num_frames=frames_to_extract):
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = 3  # Step between frames to extract
    
    # If a video does not have enough frames for consistent extraction, skip it
    if frame_count < (num_frames * interval):
        print(f"Skipping {video_path}, not enough frames.")
        cap.release()
        return 0
    
    for i in range(num_frames):
        frame_idx = i * interval
        if frame_idx >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to RGB for MTCNN/dlib consistency
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return np.array(frames) if frames else []


def detect_and_crop_face(frame):
    """Detect and crop the main face using MTCNN."""
    results = detector.detect_faces(frame)
    if not results:
        return None  # No face detected

    # Use the first detected face as the primary face
    x, y, w, h = results[0]['box']

    # Add padding around the detected face
    padding = 0.2
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(frame.shape[1] - x, w + 2 * pad_x)
    h = min(frame.shape[0] - y, h + 2 * pad_y)

    # Crop and resize to a fixed size for consistent landmark detection
    cropped_face = frame[y:y+h, x:x+w]
    if cropped_face.size == 0:
        return None

    cropped_face = cv2.resize(cropped_face, (299, 299))
    return cropped_face


def extract_landmarks(cropped_face):
    """Extract facial landmarks from the cropped face using dlib."""
    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
    rects = detector_dlib(gray_face)
    if len(rects) == 0:
        return None

    shape = predictor(gray_face, rects[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks


def compute_motion_vectors(landmarks_sequence):
    """
    Compute motion vectors (landmark deltas) between consecutive frames.
    Returns an array of shape [#frames-1, 68*2], where each row
    is the flattened (dx, dy) for the 68 landmarks.
    """
    motion_vectors = []
    valid_landmarks = [lm for lm in landmarks_sequence if lm is not None]
    
    # Need at least 2 frames to compute a delta
    if len(valid_landmarks) < 2:
        print("Not enough valid landmarks to compute motion vectors.")
        return None

    for i in range(1, len(valid_landmarks)):
        prev = valid_landmarks[i - 1]
        curr = valid_landmarks[i]
        motion = curr - prev  # (68, 2)
        motion_vectors.append(motion.flatten())  # Flatten to (136,)

    return np.array(motion_vectors)


def extract_motion_features(frames):
    """Extract motion features (landmark deltas) across the entire sequence of frames."""
    landmarks_sequence = []
    for frame in frames:
        face = detect_and_crop_face(frame)
        if face is None:
            landmarks_sequence.append(None)
            continue

        landmarks = extract_landmarks(face)
        landmarks_sequence.append(landmarks if landmarks is not None else None)

    # Only proceed if we have the full set of frames with detected landmarks
    # This can be adjusted if you want to allow partial sequences.
    if len([lm for lm in landmarks_sequence if lm is not None]) == frames_to_extract:
        motion_vectors = compute_motion_vectors(landmarks_sequence)
        return motion_vectors
    else:
        # Not enough valid landmarks across all frames
        return None


def process_videos(video_dir, label, max_videos):
    """Process videos to extract motion features and associate them with a label."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    if max_videos is not None:
        # Randomly sample up to 'max_videos' from the list
        video_files = random.sample(video_files, min(len(video_files), max_videos))
    
    motion_data = []
    labels = []
    skipped_videos = 0

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing {video_file}...")
        
        frames = extract_frames(video_path)
        # Ensure we got enough frames
        if not isinstance(frames, np.ndarray) or len(frames) < frames_to_extract:
            print(f"Skipping {video_file}: Not enough valid frames.")
            skipped_videos += 1
            continue
        
        motion_vectors = extract_motion_features(frames)
        if motion_vectors is not None:
            motion_data.append(motion_vectors)
            labels.append(label)
        else:
            skipped_videos += 1

    return motion_data, labels, skipped_videos


if __name__ == "__main__":
    # Extract motion features for real videos
    print("Extracting motion features from real videos...")
    real_motion, real_labels, real_skipped = process_videos(
        os.path.join(DATASET_PATH, 'Celeb-real'),
        label=0,
        max_videos=num_videos
    )

    # Extract motion features for fake videos
    print("Extracting motion features from fake videos...")
    fake_motion, fake_labels, fake_skipped = process_videos(
        os.path.join(DATASET_PATH, 'Celeb-synthesis'),
        label=1,
        max_videos=num_videos
    )

    # Combine and save features
    all_motion = np.array(real_motion + fake_motion, dtype=object)
    all_labels = np.array(real_labels + fake_labels)

    with open(os.path.join(FEATURES_PATH, 'motion_features.pkl'), 'wb') as f:
        pickle.dump((all_motion, all_labels), f)

    print(f"Skipped real videos: {real_skipped}")
    print(f"Skipped fake videos: {fake_skipped}")
    print("Motion delta feature extraction complete. Saved to 'features/motion_features.pkl'.")
