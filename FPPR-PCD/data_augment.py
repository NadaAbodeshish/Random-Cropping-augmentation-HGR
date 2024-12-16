import os
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

# Parameters for augmentation
num_augmentations = 3
crop_size_ratio = 0.95
brightness_range = (0.8, 1.2)
contrast_range = (0.8, 1.2)
rotation_range = (-15, 15)  
zoom_range = (1.0, 1.2)

def augment_point_cloud(points):
    """
    Apply augmentations to the point cloud frames: crop, zoom, rotation, brightness, and contrast.
    """
    # Ensure the points array is of type float32
    points = points.astype(np.float32)

    num_frames, num_points, num_features = points.shape  # Shape: (32, 512, 8)

    # Apply random crop per frame
    crop_factor = random.uniform(crop_size_ratio, 1.0)
    for frame_idx in range(num_frames):
        crop_mask = np.linalg.norm(points[frame_idx, :, :3], axis=1) <= crop_factor
        if np.any(crop_mask):  # Ensure at least some points remain
            cropped_frame = points[frame_idx][crop_mask]
            if len(cropped_frame) < num_points:
                # Pad to maintain the original shape
                pad_size = num_points - len(cropped_frame)
                pad_array = np.zeros((pad_size, num_features), dtype=np.float32)
                points[frame_idx] = np.vstack((cropped_frame, pad_array))
            else:
                points[frame_idx] = cropped_frame[:num_points]
    
    # Apply zoom
    zoom_factor = random.uniform(*zoom_range)
    points[:, :, :3] *= zoom_factor

    # Apply rotation
    rotation_angle = random.uniform(*rotation_range)
    rotation_axis = np.random.rand(3) - 0.5  # Random axis
    rotation_axis /= np.linalg.norm(rotation_axis)
    rot = R.from_rotvec(rotation_angle * np.pi / 180 * rotation_axis)
    for frame_idx in range(num_frames):
        points[frame_idx, :, :3] = rot.apply(points[frame_idx, :, :3])

    # Simulate brightness and contrast (scale and shift Z-values)
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    points[:, :, 2] = points[:, :, 2] * contrast + brightness

    return points

# Function to augment and save
def augment_dataset(input_dir, output_dir):
    """
    Traverse the input directory, augment each pts_label.npy file,
    and save the original + augmentations into the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through the directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == "pts_label.npy":  # Target only the relevant files
                file_path = os.path.join(root, file)

                # Build the corresponding output path
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Load the original point cloud
                points = np.load(file_path)

                # Save the original file
                original_output_path = os.path.join(output_subdir, "original_" + file)
                np.save(original_output_path, points)
                print(f"Saved original file: {original_output_path}")

                # Generate augmentations
                for i in range(num_augmentations):
                    augmented_points = augment_point_cloud(points.copy())
                    aug_output_path = os.path.join(output_subdir, f"aug_{i}_" + file)
                    np.save(aug_output_path, augmented_points)
                    print(f"Saved augmented file: {aug_output_path}")

# Input and output directories
input_dir = "./data/shrec17/Processed_HandGestureDataset_SHREC2017"
output_dir = "./data/shrec17/augmented-dataset"

# Run the augmentation
augment_dataset(input_dir, output_dir)
