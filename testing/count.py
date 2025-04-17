import os
import pickle

FEATURES_PATH = 'features'
splits = ['train', 'val', 'test']
files = ['motion_features.pkl', 'geometric_features.pkl']

def count_labels_in_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data, labels = pickle.load(f)
        total = len(labels)
        real_count = sum(1 for l in labels if l == 0)
        fake_count = sum(1 for l in labels if l == 1)
        return total, real_count, fake_count

for split in splits:
    print(f"\n--- {split.upper()} SPLIT ---")
    for feature_file in files:
        path = os.path.join(FEATURES_PATH, split, feature_file)
        if os.path.exists(path):
            total, real, fake = count_labels_in_pkl(path)
            print(f"{feature_file}: {total} videos (Real: {real}, Fake: {fake})")
        else:
            print(f"{feature_file}: File not found.")