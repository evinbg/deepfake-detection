import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from collections import defaultdict

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = 'saved_models/fine_tuned_xception.h5'
TEST_PATH = 'processed_images/test_test'
VIDEO_MAP_PATH = 'video_id_map.csv'
IMG_SIZE = (299, 299)
BATCH_SIZE = 40

# -------------------------------
# Load the Model
# -------------------------------
model = load_model(MODEL_PATH)

# -------------------------------
# Set up the ImageDataGenerator
# -------------------------------
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    zoom_range=(0.2, 0.2)
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -------------------------------
# Predict on All Frames
# -------------------------------
pred_probs = model.predict(test_generator, verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
filenames = test_generator.filenames  # e.g., 'real/0519_frame12.jpg'

# -------------------------------
# Load Video ID to Name Map
# -------------------------------
map_df = pd.read_csv(VIDEO_MAP_PATH)
id_to_name = dict(zip(map_df["video_id"].astype(str), map_df["filename"]))

# -------------------------------
# Group Predictions by Video ID
# -------------------------------
video_preds = defaultdict(list)
video_truth = {}

for fname, prob in zip(filenames, pred_probs.flatten()):
    base = os.path.basename(fname)
    video_id = base.split('_frame')[0]

    label_folder = os.path.normpath(fname).split(os.sep)[0].lower()
    true_label = 0 if label_folder == 'real' else 1

    video_preds[video_id].append(prob)
    video_truth[video_id] = true_label

# -------------------------------
# Evaluate All Videos
# -------------------------------
video_results = []
print("\n=== Video Predictions ===")
for video_id, probs in video_preds.items():
    avg_prob = np.mean(probs)
    pred_label = int(avg_prob > 0.5)
    true_label = video_truth[video_id]

    video_name = video_id

    print(f"{video_name} | Prob: {avg_prob * 100:.2f}% | Pred: {'Fake' if pred_label else 'Real'} | True: {'Fake' if true_label else 'Real'}")

    video_results.append({
        'video_id': video_id,
        'avg_pred_prob': avg_prob,
        'predicted_label': pred_label,
        'true_label': true_label
    })

# -------------------------------
# Save CSV
# -------------------------------
video_df = pd.DataFrame(video_results)
video_df.to_csv('xception_video_predictions.csv', index=False)
print("\nSaved video-level predictions to xception_video_predictions.csv")
