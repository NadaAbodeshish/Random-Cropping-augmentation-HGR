from pathlib import Path
from PIL import Image, ImageEnhance
import random
import shutil

# # Define input and output directories
input_dir = Path('../images_d/SHREC2017.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean/train')
output_dir = Path('../images_d_augmented/SHREC2017.mVOs-3d.28g-noisy(raw).960px-[allVOs].adaptive-mean/train')
output_dir.mkdir(parents=True, exist_ok=True)

# Parameters for augmentation
num_augmentations = 3  
crop_size_ratio = 0.95  
brightness_range = (0.8, 1.2)
contrast_range = (0.8, 1.2)
rotation_range = (-15, 15) 
zoom_range = (1.0, 1.2)  

target_orientations = {"top-down.png", "custom.png", "front-away.png"}

def augment_and_save(image, img_path, output_path, aug_index):
    # Random crop and resize
    width, height = image.size
    crop_width, crop_height = int(width * crop_size_ratio), int(height * crop_size_ratio)
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    cropped_img = image.crop((left, top, left + crop_width, top + crop_height))

    # Apply zoom by resizing with a random scale factor
    zoom_factor = random.uniform(*zoom_range)
    zoom_width, zoom_height = int(crop_width * zoom_factor), int(crop_height * zoom_factor)
    cropped_img = cropped_img.resize((zoom_width, zoom_height)).resize((width, height))

    # Random rotation
    rotation_angle = random.uniform(*rotation_range)
    cropped_img = cropped_img.rotate(rotation_angle)

    # Random brightness and contrast adjustments
    enhancer = ImageEnhance.Brightness(cropped_img)
    cropped_img = enhancer.enhance(random.uniform(*brightness_range))
    enhancer = ImageEnhance.Contrast(cropped_img)
    cropped_img = enhancer.enhance(random.uniform(*contrast_range))

    # Save with unique name
    aug_img_path = output_path / f"{img_path.stem}_aug{aug_index}.png"
    cropped_img.save(aug_img_path)

for gesture_class in input_dir.iterdir():
    if gesture_class.is_dir():
        for instance_folder in gesture_class.iterdir():
            output_instance_folder = output_dir / gesture_class.name / instance_folder.name
            output_instance_folder.mkdir(parents=True, exist_ok=True)
            
            for img_path in instance_folder.glob("*.png"):
                # Check if the image file name matches one of the target orientations
                if img_path.name in target_orientations:
                    original_img_path = output_instance_folder / f"{img_path.stem}_aug1.png"
                    shutil.copy(img_path, original_img_path)

                    with Image.open(img_path) as img:
                        for aug_index in range(2, num_augmentations + 2):  
                            augment_and_save(img, img_path, output_instance_folder, aug_index)
print("Data augmentation completed.")
