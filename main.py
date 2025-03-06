import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16 # VGG16 model for spatial feature extraction
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from mtcnn import MTCNN # MTCNN for face detection
import dlib # dlib for facial landmark detection
import pickle

# Global variables
frames_to_extract = 10
num_videos = 5

# Paths
DATASET_PATH = 'datasets/celeb'
FEATURES_PATH = 'features/'
os.makedirs(FEATURES_PATH, exist_ok=True)
CROPPED_FACES_PATH = 'cropped_faces/'
os.makedirs(CROPPED_FACES_PATH, exist_ok=True)

# Load VGG16 model for spatial feature extraction
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

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
    interval = max(1, frame_count // num_frames) # May want to change this to fixed interval
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

def detect_and_crop_face(frame, video_file, frame_idx, padding=0.2):
    """Detect and crop face using MTCNN."""
    results = detector.detect_faces(frame)
    if results:
        # Assuming the first detected face is the main one
        x, y, w, h = results[0]['box']

        # Calculate padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        # Expand the box with padding
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(frame.shape[1] - x, w + 2 * pad_x)
        h = min(frame.shape[0] - y, h + 2 * pad_y)

        # Crop and resize face
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (224, 224))

        # Save the cropped face image
        face_filename = f"{CROPPED_FACES_PATH}{video_file}_frame{frame_idx}.jpg"
        cv2.imwrite(face_filename, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

        return cropped_face
    return None

def extract_landmarks(cropped_face):
    """Extract facial landmarks using dlib."""
    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
    rects = dlib.rectangle(0, 0, 224, 224)
    shape = predictor(gray_face, rects)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks.flatten()


def extract_vgg16_features(frames, video_file):
    """Extract VGG16 spatial features and dlib landmarks for each cropped face."""
    spatial_features = []
    temporal_features = []

    for idx, frame in enumerate(frames):
        face = detect_and_crop_face(frame, video_file, idx)
        if face is not None:
            # Extract VGG16 spatial features
            x = image.img_to_array(face)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            spatial_features.append(feature.flatten())

            # Extract dlib landmarks for temporal features
            landmarks = extract_landmarks(face)
            temporal_features.append(landmarks)

    return np.array(spatial_features), np.array(temporal_features)

def process_videos(video_dir, label, max_videos):
    """Process videos to extract features and save them."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files = video_files[:max_videos]
    spatial_data = []
    temporal_data = []
    labels = []
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing {video_file}...")
        
        frames = extract_frames(video_path)
        if len(frames) < 10:
            print(f"Skipping {video_file}, not enough frames.")
            continue
        
        spatial_features, temporal_features = extract_vgg16_features(frames, video_file)
        if spatial_features.size > 0 and temporal_features.size > 0:
            spatial_data.append(spatial_features)
            temporal_data.append(temporal_features)
            labels.append(label)
    
    return spatial_data, temporal_data, labels


# Extract features for real and fake videos
print("Extracting features from real videos")
real_spatial, real_temporal, real_labels = process_videos(os.path.join(DATASET_PATH, 'Celeb-real'), label=0, max_videos=num_videos)
print("Extracting features from fake videos")
fake_spatial, fake_temporal, fake_labels = process_videos(os.path.join(DATASET_PATH, 'Celeb-synthesis'), label=1, max_videos=num_videos)

# Combine and save features
all_spatial = np.array(real_spatial + fake_spatial)
all_temporal = np.array(real_temporal + fake_temporal)
all_labels = np.array(real_labels + fake_labels)

with open(os.path.join(FEATURES_PATH, 'spatial_features.pkl'), 'wb') as f:
    pickle.dump((all_spatial, all_labels), f)

with open(os.path.join(FEATURES_PATH, 'temporal_features.pkl'), 'wb') as f:
    pickle.dump((all_temporal, all_labels), f)

print("Feature extraction complete. Saved to 'features/spatial_features.pkl'.")