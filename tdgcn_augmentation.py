import numpy as np
import os
import json
import random
from scipy.spatial.transform import Rotation as R

# Parameters for augmentation
num_augmentations = 3
crop_size_ratio = 0.95
brightness_range = (0.8, 1.2)
contrast_range = (0.8, 1.2)
rotation_range = (-15, 15)  # degrees
zoom_range = (1.0, 1.2)

# Input and output directories
input_dir = "./shrec17_jsons/train_jsons/"
output_dir = "./shrec17_jsons/train_augmented_jsons/"
os.makedirs(output_dir, exist_ok=True)

def augment_sample(skeleton_data):
    augmented_samples = []

    for _ in range(num_augmentations):
        # Random crop
        T = skeleton_data.shape[0]
        cropped_T = int(T * crop_size_ratio)
        start_idx = random.randint(0, T - cropped_T)
        skeleton_data_cropped = skeleton_data[start_idx:start_idx + cropped_T]

        # Random rotation
        rotation_angle = random.uniform(*rotation_range)
        rotation_matrix = R.from_euler('z', rotation_angle, degrees=True).as_matrix()
        skeleton_data_rotated = np.dot(skeleton_data_cropped, rotation_matrix)

        # Random zoom
        zoom_factor = random.uniform(*zoom_range)
        skeleton_data_zoomed = skeleton_data_rotated * zoom_factor

        # Random brightness and contrast (applied to coordinates)
        brightness_factor = random.uniform(*brightness_range)
        contrast_factor = random.uniform(*contrast_range)
        skeleton_data_augmented = skeleton_data_zoomed * contrast_factor + brightness_factor

        augmented_samples.append(skeleton_data_augmented)

    return augmented_samples

# Process all JSON files
for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):
        with open(os.path.join(input_dir, file_name), 'r') as f:
            sample_data = json.load(f)

        skeleton_data = np.array(sample_data['skeletons'])

        # Generate augmentations
        augmented_samples = augment_sample(skeleton_data)

        # Save augmented samples
        for idx, augmented_data in enumerate(augmented_samples):
            augmented_file_name = f"{file_name[:-5]}_aug{idx+1}.json"
            augmented_sample_data = {
                "file_name": augmented_file_name,
                "skeletons": augmented_data.tolist(),
                "label_14": sample_data['label_14'],
                "label_28": sample_data['label_28']
            }
            with open(os.path.join(output_dir, augmented_file_name), 'w') as aug_f:
                json.dump(augmented_sample_data, aug_f)

        # Copy original sample to the augmented directory
        shutil.copy(os.path.join(input_dir, file_name), os.path.join(output_dir, file_name))

print("Data augmentation complete!")
