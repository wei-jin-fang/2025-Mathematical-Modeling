import cv2
import numpy as np
import os
import shutil
from pathlib import Path


def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def draw_random_lines(image, num_lines=5):
    """Draw random light-colored lines on the image."""
    img_copy = image.copy()
    height, width = img_copy.shape[:2]
    for _ in range(num_lines):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        color = (np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255))
        thickness = np.random.randint(1, 3)
        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy


def augment_image(image, augmentation_type):
    """Apply specified augmentation to the image."""
    if augmentation_type == 'noise':
        return add_gaussian_noise(image)
    elif augmentation_type == 'lines':
        return draw_random_lines(image)
    elif augmentation_type == 'noise_lines':
        noisy = add_gaussian_noise(image)
        return draw_random_lines(noisy)
    return image


def process_images(input_image_dir, input_mask_dir, output_image_dir, output_mask_dir, num_augmentations):
    """Process images and masks for augmentation."""
    # Create output directories if they don't exist
    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)

    # Define augmentation types
    augmentation_types = ['noise', 'lines', 'noise_lines']

    # Get list of images
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.jpg')]

    for img_file in image_files:
        # Read image
        img_path = os.path.join(input_image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Read corresponding mask
        mask_file = img_file.replace('.jpg', '_mask_mask.png')
        mask_path = os.path.join(input_mask_dir, mask_file)
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue

        # Generate augmentations
        for i in range(num_augmentations):
            # Select augmentation type
            aug_type = augmentation_types[i % len(augmentation_types)]

            # Apply augmentation
            augmented_image = augment_image(image, aug_type)

            # Generate output filenames
            base_name = img_file.split('.')[0]
            aug_img_name = f"{base_name}_{i + 1}.jpg"
            aug_mask_name = f"{base_name}_{i + 1}_mask_mask.png"

            # Save augmented image
            cv2.imwrite(os.path.join(output_image_dir, aug_img_name), augmented_image)

            # Copy original mask
            shutil.copy(mask_path, os.path.join(output_mask_dir, aug_mask_name))

            print(f"Generated: {aug_img_name} and {aug_mask_name}")


def main():
    # Define directories
    test_image_dir = './data/image/test'
    test_mask_dir = './data/mask/test'  # Updated to point to mask/test
    train_image_dir = './data/image/train'
    train_mask_dir = './data/mask/train'
    val_image_dir = './data/image/val'
    val_mask_dir = './data/mask/val'

    # Process for train set (8 augmentations per image)
    print("Generating augmentations for train set...")
    process_images(test_image_dir, test_mask_dir, train_image_dir, train_mask_dir, num_augmentations=8)

    # Process for val set (2 augmentations per image)
    print("Generating augmentations for val set...")
    process_images(test_image_dir, test_mask_dir, val_image_dir, val_mask_dir, num_augmentations=2)


if __name__ == "__main__":
    main()