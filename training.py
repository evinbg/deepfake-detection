import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths to preprocessed images
REAL_PATH = 'processed_images/real'
FAKE_PATH = 'processed_images/fake'

# Model parameters
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 10

# Calculate class weights
real_count = len(os.listdir(REAL_PATH))
fake_count = len(os.listdir(FAKE_PATH))
class_weights = {0: fake_count / real_count, 1: 1.0}  # 0 = real, 1 = fake

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    'processed_images/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'processed_images/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Load pre-trained Xception model without top layers
base_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))
base_model.trainable = False # Freeze base model layers

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output_layer) # Final model

model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train only the classification head (Phase 1)
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, class_weight=class_weights)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[-50:]: # Unfreeze the last 50 layers
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save the fine-tuned model
model.save('fine_tuned_xception.h5')