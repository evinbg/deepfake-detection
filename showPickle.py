# show_class_ratio.py

import pickle
import numpy as np

def show_class_ratio(motion_features_path: str):
    """
    Loads motion features and labels from the specified pickle file,
    then prints the count and percentage of Real vs. Fake samples.
    """
    with open(motion_features_path, 'rb') as f:
        (all_motion, all_labels) = pickle.load(f)

    total_samples = len(all_labels)
    num_real = np.sum(all_labels == 0)
    num_fake = np.sum(all_labels == 1)

    print(f"Total samples: {total_samples}")
    print(f"Real samples: {num_real} ({num_real / total_samples * 100:.2f}%)")
    print(f"Fake samples: {num_fake} ({num_fake / total_samples * 100:.2f}%)")

if __name__ == "__main__":
    # Adjust the path to match the location of your file:
    show_class_ratio("features/motion_features.pkl")
