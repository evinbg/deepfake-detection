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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths to preprocessed images
REAL_PATH = 'processed_images/real'
FAKE_PATH = 'processed_images/fake'
IMAGES_PATH = 'processed_images'

# Model parameters
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# Create a new temporary directory for selected images
selected_path = 'processed_images_selected'
os.makedirs(selected_path, exist_ok=True)

# Create the necessary subdirectories for 'real' and 'fake'
selected_real_path = os.path.join(selected_path, 'real')
selected_fake_path = os.path.join(selected_path, 'fake')
os.makedirs(selected_real_path, exist_ok=True)
os.makedirs(selected_fake_path, exist_ok=True)

'''
MAX_IMAGES = 15000 # Set the number of images to train on

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
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=(0.1, 0.2),
    brightness_range=[0.8, 1.2],
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

# Load pre-trained Xception model without top layers
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze initial layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output_layer) # Final model

#model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision()])

# Early stopping to prevent overtraining
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, lr_scheduler]
)

def unfreeze_and_train(model, num_layers_to_unfreeze, train_generator, val_generator, epochs, lr):
    """ Unfreezes 'num_layers_to_unfreeze' layers from the base model and retrains it. """
    # Unfreeze only the last 'num_layers_to_unfreeze' layers
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    # Compile with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision()])

    # Train again with unfrozen layers
    history_finetune = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=[early_stopping, lr_scheduler]
    )

    return model, history_finetune

# Unfreeze 30 layers -> Train
model, history1 = unfreeze_and_train(model, num_layers_to_unfreeze=30, train_generator=train_generator, val_generator=val_generator, epochs=5, lr=0.0001)

# Unfreeze 60 layers -> Train
model, history2 = unfreeze_and_train(model, num_layers_to_unfreeze=60, train_generator=train_generator, val_generator=val_generator, epochs=5, lr=0.00005)

# # Unfreeze the entire model -> Train
# model, history3 = unfreeze_and_train(model, num_layers_to_unfreeze=len(base_model.layers), train_generator=train_generator, val_generator=val_generator, epochs=5, lr=0.00001)

# Evaluate Model
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    selected_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_metrics = model.evaluate(test_generator)
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")

# Save the fine-tuned model
model.save('fine_tuned_xception.h5')