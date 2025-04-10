import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths to split datasets
TRAIN_PATH = 'processed_images_new/train'
VAL_PATH = 'processed_images_new/val'
TEST_PATH = 'processed_images_new/test'

# Model parameters
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

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Load validation data
val_generator = val_test_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Load test data
test_generator = val_test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Load pre-trained Xception model without top layers
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze initial layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output_layer) # Final model

#model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision()]
)

# Early stopping to prevent overtraining
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, 
    patience=2, 
    min_lr=1e-6,
    verbose=1
)

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
model, history1 = unfreeze_and_train(
    model, 
    num_layers_to_unfreeze=30, 
    train_generator=train_generator, 
    val_generator=val_generator, 
    epochs=5, 
    lr=0.0001
)

model.save('xception_stage1_unfreeze30.h5')

# Unfreeze 60 layers -> Train
model, history2 = unfreeze_and_train(
    model, 
    num_layers_to_unfreeze=60, 
    train_generator=train_generator, 
    val_generator=val_generator, 
    epochs=5, 
    lr=0.00007
)

model.save('xception_stage2_unfreeze60.h5')

# # Unfreeze the entire model -> Train
# model, history3 = unfreeze_and_train(model, num_layers_to_unfreeze=len(base_model.layers), train_generator=train_generator, val_generator=val_generator, epochs=5, lr=0.00001)

# Save the fine-tuned model
model.save('fine_tuned_xception.h5')

test_metrics = model.evaluate(test_generator, return_dict=True)
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

# Get predictions and true labels
pred_probs = model.predict(test_generator, verbose=1)  # Probabilities
pred_labels = (pred_probs > 0.5).astype(int).flatten() # Binary classification

true_labels = test_generator.classes                   # Ground truth labels
filenames = test_generator.filenames                   # Corresponding filenames

# Save to CSV
results_df = pd.DataFrame({
    'filename': filenames,
    'true_label': true_labels,
    'predicted_prob': pred_probs.flatten(),
    'predicted_label': pred_labels
})

results_df.to_csv('xception_test_predictions.csv', index=False)

print("Predictions saved to xception_test_predictions.csv")