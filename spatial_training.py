import os
import numpy as np
import tensorflow as tf
import shutil
import random
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Paths to preprocessed images
REAL_PATH = 'processed_images/real'
FAKE_PATH = 'processed_images/fake'
IMAGES_PATH = 'processed_images'

# Model parameters
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 10

# Create a new temporary directory for selected images
selected_path = 'processed_images_selected'
os.makedirs(selected_path, exist_ok=True)

# Create the necessary subdirectories for 'real' and 'fake'
selected_real_path = os.path.join(selected_path, 'real')
selected_fake_path = os.path.join(selected_path, 'fake')
os.makedirs(selected_real_path, exist_ok=True)
os.makedirs(selected_fake_path, exist_ok=True)

'''
MAX_IMAGES = 2000 # Set the number of images to train on

# List and limit the number of images from each class
real_images = [f for f in os.listdir(REAL_PATH) if os.path.isfile(os.path.join(REAL_PATH, f))]
fake_images = [f for f in os.listdir(FAKE_PATH) if os.path.isfile(os.path.join(FAKE_PATH, f))]

# Randomly shuffle the lists
random.shuffle(real_images)
random.shuffle(fake_images)

# Select up to MAX_IMAGES randomly from each class
real_images = real_images[:min(MAX_IMAGES, len(real_images))]
fake_images = fake_images[:min(MAX_IMAGES, len(fake_images))]

# Limit to MAX_IMAGES
real_images = real_images[:MAX_IMAGES]
fake_images = fake_images[:MAX_IMAGES]

# Copy the selected images to the new directory structure
for img in real_images:
    shutil.copy(os.path.join(REAL_PATH, img), os.path.join(selected_real_path, img))

for img in fake_images:
    shutil.copy(os.path.join(FAKE_PATH, img), os.path.join(selected_fake_path, img))
'''

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=(0.1, 0.2),
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    selected_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    selected_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

#print(f"Selected {len(real_images)} real images and {len(fake_images)} fake images")

# Load pre-trained Xception model without top layers
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze initial layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x)
x = Dropout(0.6)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output_layer) # Final model

#model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overtraining
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping]
)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-50:]: # Unfreeze the last layers
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.000005), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Evaluate Model
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    selected_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the fine-tuned model
model.save('fine_tuned_xception.h5')