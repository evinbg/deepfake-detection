import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = 'xception_stage1_unfreeze20.h5'
TEST_PATH = 'processed_images_new/test'
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# Load model
model = load_model(MODEL_PATH)

# Preprocessing for test set
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

MAX_SAMPLES = 5000
steps = MAX_SAMPLES // BATCH_SIZE  # Total batches needed

# Evaluate model
#test_metrics = model.evaluate(test_generator, steps=steps, return_dict=True)
test_metrics = model.evaluate(test_generator, return_dict=True)
print(f"Evaluation for test set at: {TEST_PATH}")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# Predict
pred_probs = model.predict(test_generator, verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = test_generator.classes
filenames = test_generator.filenames

print("Predicted class distribution:", np.bincount(pred_labels))

# Save predictions to CSV
results_df = pd.DataFrame({
    'filename': filenames,
    'true_label': true_labels,
    'predicted_prob': pred_probs.flatten(),
    'predicted_label': pred_labels
})
results_df.to_csv('xception_test_predictions.csv', index=False)
print("Saved predictions to xception_test_predictions.csv")