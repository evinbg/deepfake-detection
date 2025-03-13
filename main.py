import os
import cv2
import numpy as np
from tensorflow.keras.applications import Xception # Xception model for spatial feature extraction
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
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

# Load Xception model for spatial feature extraction
base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
x = base_model.output

# Add custom layers for fine-tuning
x = Dense(512, activation='relu')(x)  # Add a fully connected layer
x = Dropout(0.5)(x)  # Dropout for regularization
x = Dense(1, activation='sigmoid')(x)  # Binary classification (real or fake)

model = Model(inputs=base_model.input, outputs=x) # Final model

# Freeze the base layers initially to retain the pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

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
    interval = 10
    
    if frame_count < 10:  # Handle videos with fewer than 10 frames
        return 0
    
    for i in range(num_frames):
        frame_idx = i * interval
        if frame_idx >= frame_count:  # Stop if we exceed the total frame count
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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

        # Crop and resize face (299x299 for Xception)
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (299, 299))

        # Save the cropped face image
        face_filename = f"{CROPPED_FACES_PATH}{video_file}_frame{frame_idx}.jpg"
        cv2.imwrite(face_filename, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

        return cropped_face
    return None

def extract_landmarks(cropped_face):
    """Extract facial landmarks using dlib."""
    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
    rects = dlib.rectangle(0, 0, 299, 299)
    shape = predictor(gray_face, rects)
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
    for i in range(1, len(landmarks_sequence)):
        prev = landmarks_sequence[i - 1]
        curr = landmarks_sequence[i]
        motion = curr - prev  # Calculate delta (dx, dy) for each point
        motion_vectors.append(motion.flatten())  # Flatten to a single array
    return np.array(motion_vectors)

def extract_features(frames, video_file):
    """Extract Xception spatial features and dlib landmarks for each cropped face."""
    spatial_features = []
    landmarks_sequence = []
    geometric_features = []

    for idx, frame in enumerate(frames):
        face = detect_and_crop_face(frame, video_file, idx)
        if face is not None:
            # Extract spatial features
            x = image.img_to_array(face)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            spatial_features.append(feature.flatten())

            # Extract dlib landmarks for temporal features
            landmarks = extract_landmarks(face)
            landmarks_sequence.append(landmarks)

            # Compute geometric features (e.g., distances between eyes, mouth, etc.)
            geo_features = compute_geometric_features(landmarks)
            geometric_features.append(geo_features)

    if len(landmarks_sequence) == frames_to_extract:
        # Compute motion vectors if all frames have landmarks
        motion_vectors = compute_motion_vectors(landmarks_sequence)
        return np.array(spatial_features), motion_vectors, np.array(geometric_features)
    else:
        return np.array(spatial_features), None, None

def process_videos(video_dir, label, max_videos):
    """Process videos to extract features and save them."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files = video_files[:max_videos]
    spatial_data = []
    motion_data = []
    geometric_data = []
    labels = []
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing {video_file}...")
        
        frames = extract_frames(video_path)
        if len(frames) < 10:
            print(f"Skipping {video_file}, not enough frames.")
            continue
        
        spatial_features, motion_vectors, geometric_features = extract_features(frames, video_file)
        if spatial_features.size > 0 and motion_vectors is not None and geometric_features is not None:
            spatial_data.append(spatial_features)
            motion_data.append(motion_vectors)
            geometric_data.append(geometric_features)
            labels.append(label)
    
    return spatial_data, motion_data, geometric_data, labels


# Extract features for real and fake videos
print("Extracting features from real videos")
real_spatial, real_motion, real_geometric, real_labels = process_videos(os.path.join(DATASET_PATH, 'Celeb-real'), label=0, max_videos=num_videos)
print("Extracting features from fake videos")
fake_spatial, fake_motion, fake_geometric, fake_labels = process_videos(os.path.join(DATASET_PATH, 'Celeb-synthesis'), label=1, max_videos=num_videos)

# Combine and save features
all_spatial = np.array(real_spatial + fake_spatial)
all_motion = np.array(real_motion + fake_motion)
all_geometric = np.array(real_geometric + fake_geometric)
all_labels = np.array(real_labels + fake_labels)

with open(os.path.join(FEATURES_PATH, 'spatial_features.pkl'), 'wb') as f:
    pickle.dump((all_spatial, all_labels), f)

with open(os.path.join(FEATURES_PATH, 'motion_features.pkl'), 'wb') as f:
    pickle.dump((all_motion, all_labels), f)

with open(os.path.join(FEATURES_PATH, 'geometric_features.pkl'), 'wb') as f:
    pickle.dump((all_geometric, all_labels), f)

print("Feature extraction complete. Saved to 'features/spatial_features.pkl', 'features/motion_features.pkl', and 'features/geometric_features.pkl'.")

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
    layer.trainable = True

# Recompile the model after unfreezing
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model (train the model on dataset)
history = model.fit(
    all_spatial, all_labels,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Save the fine-tuned model
model.save('fine_tuned_xception.h5')