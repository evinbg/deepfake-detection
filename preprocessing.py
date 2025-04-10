import os
import cv2
import numpy as np
from mtcnn import MTCNN # MTCNN for face detection
import dlib # dlib for facial landmark detection
import pickle
import random

# Global variables
frames_to_extract = 40
num_videos = 590

# Paths
DATASET_PATH = 'datasets/celeb'
FEATURES_PATH = 'features/'
os.makedirs(FEATURES_PATH, exist_ok=True)
PROCESSED_IMAGES_PATH = 'processed_images/'
os.makedirs(PROCESSED_IMAGES_PATH, exist_ok=True)
REAL_PATH = os.path.join(PROCESSED_IMAGES_PATH, 'real')
FAKE_PATH = os.path.join(PROCESSED_IMAGES_PATH, 'fake')
os.makedirs(REAL_PATH, exist_ok=True)
os.makedirs(FAKE_PATH, exist_ok=True)

# Initialize MTCNN for face detection
detector = MTCNN()

# Load dlib's face detector
detector_dlib = dlib.get_frontal_face_detector()

# Load dlib's pre-trained facial landmark predictor
predictor_path = '/Users/djamelalmabouada/Desktop/capstone_project/deepfake-detection/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)


def extract_frames(video_path, num_frames=frames_to_extract):
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = 3
    
    if frame_count < (frames_to_extract * interval): # Handle videos with fewer than the required amount of frames
        print(f"Skipping {video_path}, not enough frames.")
        return 0
    
    for i in range(num_frames):
        frame_idx = i * interval
        if frame_idx >= frame_count: # Stop if we exceed the total frame count
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return np.array(frames) if frames else []

def detect_and_crop_face(frame, video_file, frame_idx, label):
    """Detect and crop face using MTCNN."""
    results = detector.detect_faces(frame)

    if not results:
        return None # Return None if no face is detected

    # Assuming the first detected face is the main one
    x, y, w, h = results[0]['box']

    # Calculate padding
    padding = 0.2
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    # Expand the box with padding
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(frame.shape[1] - x, w + 2 * pad_x)
    h = min(frame.shape[0] - y, h + 2 * pad_y)

    # Crop and resize face (299x299 for Xception)
    cropped_face = frame[y:y+h, x:x+w]

    if cropped_face.size == 0:
        return None # Ensure valid cropped face

    cropped_face = cv2.resize(cropped_face, (299, 299))

    # Determine the correct folder
    if label == 0: # Real
        folder = REAL_PATH
    else: # Fake
        folder = FAKE_PATH

    # Save the cropped face image
    face_filename = os.path.join(folder, f"{video_file}_frame{frame_idx}.jpg")
    cv2.imwrite(face_filename, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

    return cropped_face

def extract_landmarks(cropped_face):
    """Extract facial landmarks using dlib."""
    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
    rects = detector_dlib(gray_face)

    if len(rects) == 0:
        return None # Return None if no face is detected

    shape = predictor(gray_face, rects[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def compute_geometric_features(landmarks):
    """Compute geometric distances between key facial landmarks."""
    # Reshape flattened landmarks into (68, 2) array
    points = landmarks.reshape((68, 2))
    
    eye_distance = np.linalg.norm(points[36] - points[45])  # Eye corners
    mouth_width = np.linalg.norm(points[48] - points[54])   # Mouth corners
    nose_to_chin = np.linalg.norm(points[30] - points[8])   # Nose tip to chin
    
    # Return as a feature vector
    return np.array([eye_distance, mouth_width, nose_to_chin])

def compute_motion_vectors(landmarks_sequence):
    """Compute motion vectors from a sequence of landmarks."""
    motion_vectors = []

    valid_landmarks = [l for l in landmarks_sequence if l is not None]
    if len(valid_landmarks) < 2:
        print("Not enough valid landmarks to compute motion vectors.")
        return None # Not enough valid landmarks to compute motion vectors

    for i in range(1, len(valid_landmarks)):
        prev = valid_landmarks[i - 1]
        curr = valid_landmarks[i]
        motion = curr - prev # Calculate delta (dx, dy) for each point
        motion_vectors.append(motion.flatten()) # Flatten to a single array
    
    return np.array(motion_vectors)

def extract_features(frames, video_file, label):
    """Extract dlib landmarks for each cropped face."""
    landmarks_sequence = []
    geometric_features = []

    for idx, frame in enumerate(frames):
        face = detect_and_crop_face(frame, video_file, idx, label)
        if face is None:
            print(f"Skipping frame {idx} in {video_file}: No face detected.")
            continue # Skip frames where no face was detected

        landmarks = extract_landmarks(face) # Extract dlib landmarks
        if landmarks is None:
            print(f"Skipping frame {idx} in {video_file}: No landmarks detected.")
            continue # Skip if no landmarks detected

        landmarks_sequence.append(landmarks)

        # Compute geometric features (e.g., distances between eyes, mouth, etc.)
        geo_features = compute_geometric_features(landmarks)

        if geo_features is not None:
            geometric_features.append(geo_features)

    if len(landmarks_sequence) == frames_to_extract:
        # Compute motion vectors if all frames have landmarks
        motion_vectors = compute_motion_vectors(landmarks_sequence)
        return motion_vectors, np.array(geometric_features)
    else:
        print(f"Skipping {video_file}: Not enough valid frames for feature extraction.")
        return None, None

def process_videos(video_dir, label, max_videos):
    """Process videos to extract features and save them."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    if max_videos is not None:
        # Randomly select num_amount of fake videos from the list
        video_files = random.sample(video_files, min(len(video_files), max_videos))
    
    motion_data = []
    geometric_data = []
    labels = []
    skipped_videos = 0 # Track skipped videos
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing {video_file}...")
        
        frames = extract_frames(video_path)
        if not isinstance(frames, np.ndarray) or len(frames) < frames_to_extract:
            print(f"Skipping {video_file}: Not enough valid frames.")
            skipped_videos += 1
            continue
        
        motion_vectors, geometric_features = extract_features(frames, video_file, label)
        if motion_vectors is not None and geometric_features is not None:
            motion_data.append(motion_vectors)
            geometric_data.append(geometric_features)
            labels.append(label)
        else:
            skipped_videos += 1
    
    return motion_data, geometric_data, labels, skipped_videos


# Extract features for real and fake videos
print("Extracting features from real videos")
real_motion, real_geometric, real_labels, real_skipped = process_videos(os.path.join(DATASET_PATH, 'Celeb-real'), label=0, max_videos=num_videos)
print("Extracting features from fake videos")
fake_motion, fake_geometric, fake_labels, fake_skipped = process_videos(os.path.join(DATASET_PATH, 'Celeb-synthesis'), label=1, max_videos=num_videos)

# Combine and save features
all_motion = np.array(real_motion + fake_motion)
all_geometric = np.array(real_geometric + fake_geometric)
all_labels = np.array(real_labels + fake_labels)

with open(os.path.join(FEATURES_PATH, 'motion_features.pkl'), 'wb') as f:
    pickle.dump((all_motion, all_labels), f)

with open(os.path.join(FEATURES_PATH, 'geometric_features.pkl'), 'wb') as f:
    pickle.dump((all_geometric, all_labels), f)

print(f"Skipped real videos: {real_skipped}")
print(f"Skipped fake videos: {fake_skipped}")
print("Feature extraction complete. Saved to 'features/motion_features.pkl', and 'features/geometric_features.pkl'.")
