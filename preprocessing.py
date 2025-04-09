import os
import cv2
import numpy as np
from mtcnn import MTCNN # MTCNN for face detection
import dlib # dlib for facial landmark detection
import pickle
import random
import shutil
from sklearn.model_selection import train_test_split
import csv

# Global variables
frames_to_extract = 40
num_videos = 590
interval = 3

# To store the mapping between ID and original video file
video_id_map = []

# Paths
DATASET_PATH = 'datasets/celeb'
FEATURES_PATH = 'features/'
PROCESSED_IMAGES_PATH = 'processed_images_new/'

REAL_PATH = os.path.join(PROCESSED_IMAGES_PATH, 'real')
FAKE_PATH = os.path.join(PROCESSED_IMAGES_PATH, 'fake')

os.makedirs(FEATURES_PATH, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_PATH, exist_ok=True)
os.makedirs(REAL_PATH, exist_ok=True)
os.makedirs(FAKE_PATH, exist_ok=True)

# Initialize MTCNN for face detection
detector = MTCNN()

# Load dlib's face detector
detector_dlib = dlib.get_frontal_face_detector()

# Load dlib's pre-trained facial landmark predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)


def extract_frames(video_path, num_frames=frames_to_extract):
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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

def detect_and_crop_face(frame, video_id, frame_idx, label):
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
    label_str = 'real' if label == 0 else 'fake'
    folder = os.path.join(PROCESSED_IMAGES_PATH, label_str)
    face_filename = os.path.join(folder, f"{video_id}_frame{frame_idx}.jpg")
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

def compute_geometric_features(landmarks_sequence):
    """Compute geometric distances between key facial landmarks."""
    geometric_features = []

    for landmarks in landmarks_sequence:
        if landmarks is None or landmarks.shape != (68, 2):
            continue # skip invalid landmarks

        eye_distance = np.linalg.norm(landmarks[36] - landmarks[45]) # Eye corners
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])  # Mouth corners
        nose_to_chin = np.linalg.norm(landmarks[30] - landmarks[8])  # Nose tip to chin

        feature_vector = np.array([eye_distance, mouth_width, nose_to_chin])
        geometric_features.append(feature_vector)

    return np.array(geometric_features)

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

def extract_features(frames, video_id, label):
    """Extract dlib landmarks for each cropped face."""
    landmarks_sequence = []

    for idx, frame in enumerate(frames):
        face = detect_and_crop_face(frame, video_id, idx, label)
        if face is None:
            print(f"Skipping frame {idx} in {video_id}: No face detected.")
            return None, None

        landmarks = extract_landmarks(face) # Extract dlib landmarks
        if landmarks is None:
            print(f"Skipping {video_id}: No landmarks detected in frame {idx}.")
            return None, None

        landmarks_sequence.append(landmarks)

    if len(landmarks_sequence) == frames_to_extract:
        # Compute if all frames have landmarks
        geo_features = compute_geometric_features(landmarks_sequence)
        motion_vectors = compute_motion_vectors(landmarks_sequence)
        return motion_vectors, geo_features
    else:
        return None, None

def process_videos(video_dir, label, max_videos, id_offset=0):
    """Process videos to extract features and save them."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files = random.sample(video_files, min(len(video_files), max_videos))

    motion_data, geometric_data, labels, video_ids = [], [], [], []
    skipped_videos = 0 # Track skipped videos
    
    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        video_id = f"{id_offset + idx:04d}"
        print(f"Processing {video_file} as ID {video_id}...")
        
        frames = extract_frames(video_path)
        if not isinstance(frames, np.ndarray) or len(frames) < frames_to_extract:
            print(f"Skipping {video_file}: Not enough valid frames.")
            skipped_videos += 1
            print(f"Skipped: {skipped_videos}")
            continue
        
        motion_vectors, geometric_features = extract_features(frames, video_id, label)
        if motion_vectors is not None and geometric_features is not None:
            motion_data.append(motion_vectors)
            geometric_data.append(geometric_features)
            labels.append(label)
            video_ids.append(video_id)
            video_id_map.append({
                "video_id": video_id,
                "filename": video_file,
                "label": "real" if label == 0 else "fake"
            })
        else:
            skipped_videos += 1
            print(f"Skipped: {skipped_videos}")
            continue
    
    return motion_data, geometric_data, labels, video_ids, skipped_videos

def save_split_features(motion, geometric, labels, indices, split_name):
    split_motion = [motion[i] for i in indices]
    split_geometric = [geometric[i] for i in indices]
    split_labels = [labels[i] for i in indices]

    split_path = os.path.join(FEATURES_PATH, split_name)
    os.makedirs(split_path, exist_ok=True)

    with open(os.path.join(split_path, 'motion_features.pkl'), 'wb') as f:
        pickle.dump((np.array(split_motion), np.array(split_labels)), f)

    with open(os.path.join(split_path, 'geometric_features.pkl'), 'wb') as f:
        pickle.dump((np.array(split_geometric), np.array(split_labels)), f)

def move_cropped_images(video_ids, labels, split_name):
    for video_id, label in zip(video_ids, labels):
        label_str = 'real' if label == 0 else 'fake'
        src_dir = os.path.join(PROCESSED_IMAGES_PATH, label_str)
        dst_dir = os.path.join(PROCESSED_IMAGES_PATH, split_name, label_str)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            if fname.startswith(video_id):
                shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))


# Extract features for real and fake videos
print("Extracting features from real videos")
real_motion, real_geometric, real_labels, real_ids, real_skipped = process_videos(
    os.path.join(DATASET_PATH, 'Celeb-real'), 
    label=0, 
    max_videos=num_videos
)
real_valid = len(real_motion)
print(f"Skipped real videos: {real_skipped}")

# Extract features from fake videos to match the number of valid real videos
print("Extracting features from fake videos to match valid real videos")
fake_motion, fake_geometric, fake_labels, fake_ids = [], [], [], []
fake_skipped_total = 0

fake_video_dir = os.path.join(DATASET_PATH, 'Celeb-synthesis')
all_fake_videos = [f for f in os.listdir(fake_video_dir) if f.endswith('.mp4')]
random.shuffle(all_fake_videos)

fake_idx = 0
while len(fake_motion) < real_valid and fake_idx < len(all_fake_videos):
    remaining_needed = real_valid - len(fake_motion)
    batch_videos = all_fake_videos[fake_idx:fake_idx + remaining_needed]

    motion, geometric, labels, ids, skipped = process_videos(
        fake_video_dir,
        label=1,
        max_videos=len(batch_videos),
        id_offset=30000 + fake_idx
    )

    fake_motion.extend(motion)
    fake_geometric.extend(geometric)
    fake_labels.extend(labels)
    fake_ids.extend(ids)
    fake_skipped_total += skipped
    fake_idx += len(batch_videos)

fake_valid = len(fake_motion)
print(f"Target fake videos: {real_valid}")
print(f"Collected fake videos: {fake_valid}")
print(f"Skipped fake videos (total): {fake_skipped_total}")

# Combine and save features
all_motion = real_motion + fake_motion
all_geometric = real_geometric + fake_geometric
all_labels = real_labels + fake_labels
all_ids = real_ids + fake_ids

# Train/val/test split
num_samples = len(all_labels)
indices = list(range(num_samples))
random.shuffle(indices)

train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# Save features
save_split_features(all_motion, all_geometric, all_labels, train_indices, "train")
save_split_features(all_motion, all_geometric, all_labels, val_indices, "val")
save_split_features(all_motion, all_geometric, all_labels, test_indices, "test")

# Move images
move_cropped_images([all_ids[i] for i in train_indices], [all_labels[i] for i in train_indices], "train")
move_cropped_images([all_ids[i] for i in val_indices], [all_labels[i] for i in val_indices], "val")
move_cropped_images([all_ids[i] for i in test_indices], [all_labels[i] for i in test_indices], "test")

# Final stats
print(f"Final dataset sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
print(f"Skipped real videos: {real_skipped}")
print(f"Skipped fake videos: {fake_skipped_total}")

print(f"Total valid real videos: {len(real_motion)}")
print(f"Total valid fake videos: {len(fake_motion)}")

# Save video ID and filename mapping
with open(os.path.join(FEATURES_PATH, 'video_id_map.csv'), 'w', newline='') as csvfile:
    fieldnames = ['video_id', 'filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for entry in video_id_map:
        writer.writerow(entry)

print(f"Video ID mapping saved to {os.path.join(FEATURES_PATH, 'video_id_map.csv')}")