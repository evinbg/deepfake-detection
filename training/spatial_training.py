import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.spatial_model import SpatialXceptionFT

# Paths
TRAIN_PATH = '../processed_images_new/train'
VAL_PATH = '../processed_images_new/val'
TEST_PATH = '../processed_images_new/test'
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=(0.2, 0.25),
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    zoom_range=(0.2, 0.2)
)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
)
val_generator = val_test_datagen.flow_from_directory(
    VAL_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)
test_generator = val_test_datagen.flow_from_directory(
    TEST_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# Build and compile model
model_builder = SpatialXceptionFT()
model = model_builder.build_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision()]
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

# Initial training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, lr_scheduler]
)

# Fine-tuning
def unfreeze_and_train(model, num_layers_to_unfreeze, train_generator, val_generator, epochs, lr):
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision()]
    )

    history_finetune = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=[early_stopping, lr_scheduler]
    )
    return model, history_finetune

# Unfreeze and retrain
model, history1 = unfreeze_and_train(
    model, num_layers_to_unfreeze=20,
    train_generator=train_generator,
    val_generator=val_generator,
    epochs=10,
    lr=0.0001
)

# Save model
model.save('fine_tuned_xception.h5')