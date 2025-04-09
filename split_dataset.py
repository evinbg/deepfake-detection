import os
import shutil
import random

# Input and output paths
SOURCE_DIR = 'processed_images_selected'
OUTPUT_BASE = 'split_images'

SPLITS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

CLASSES = ['real', 'fake']

# Paths to preprocessed images
REAL_PATH = 'processed_images/real'
FAKE_PATH = 'processed_images/fake'
IMAGES_PATH = 'processed_images'

# Paths to split datasets
TRAIN_PATH = 'split_images/train'
VAL_PATH = 'split_images/val'
TEST_PATH = 'split_images/test'

# Create a new directory for selected images
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

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def split_and_copy_images(class_name):
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]

    random.shuffle(images)

    total = len(images)
    num_train = int(total * SPLITS['train'])
    num_val = int(total * SPLITS['val'])
    num_test = total - num_train - num_val # Ensure total counts add up

    split_data = {
        'train': images[:num_train],
        'val': images[num_train:num_train + num_val],
        'test': images[num_train + num_val:]
    }

    for split, split_images in split_data.items():
        split_class_dir = os.path.join(OUTPUT_BASE, split, class_name)
        create_dir(split_class_dir)

        for img in split_images:
            src = os.path.join(source_class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)

    print(f"Split {class_name} images -> train: {num_train}, val: {num_val}, test: {num_test}")

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory '{SOURCE_DIR}' does not exist.")
        return

    for class_name in CLASSES:
        split_and_copy_images(class_name)

    print("Dataset split completed.")

if __name__ == "__main__":
    main()