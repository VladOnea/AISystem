from PIL import Image, ImageEnhance
import random
import os
import glob
import numpy as np

def mirrorImage(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rotateImage(image, angle):
    return image.rotate(angle)

def translateImage(image, max_translate):
    x_translate = random.randint(-max_translate, max_translate)
    y_translate = random.randint(-max_translate, max_translate)
    return image.transform(image.size, Image.AFFINE, (1, 0, x_translate, 0, 1, y_translate))

def scaleImage(image, scale_factor):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def adjustBrightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def augmentImage(image, filename):
    augmented_images = []
    filenames = []

    mirrored_img = mirrorImage(image)
    augmented_images.append(mirrored_img)
    filenames.append(f"mirrored_{filename}")

    angles = [90, 180, 270]
    for i, angle in enumerate(angles):
        rotated_img = rotateImage(image, angle)
        augmented_images.append(rotated_img)
        filenames.append(f"rotated_{angle}_{filename}")

    max_translate = 10
    translated_img = translateImage(image, max_translate)
    augmented_images.append(translated_img)
    filenames.append(f"translated_{filename}")

    scale_factors = [0.9, 1.1]
    for i, scale_factor in enumerate(scale_factors):
        scaled_img = scaleImage(image, scale_factor)
        augmented_images.append(scaled_img)
        filenames.append(f"scaled_{scale_factor}_{filename}")

    brightness_factors = [0.8, 1.2]
    for i, factor in enumerate(brightness_factors):
        bright_img = adjustBrightness(image, factor)
        augmented_images.append(bright_img)
        filenames.append(f"brightness_{factor}_{filename}")

    return augmented_images, filenames

def loadImages(directory_path, pattern='*.bmp'):
    image_pattern = os.path.join(directory_path, pattern)
    images = []
    filenames = []

    for image_file in glob.glob(image_pattern):
        try:
            with Image.open(image_file) as img:
                images.append(img.copy())
                base_filename = os.path.basename(image_file)
                filenames.append(base_filename) 
                augmented_images, augmented_filenames = augmentImage(img, base_filename)
                images.extend(augmented_images)
                filenames.extend(augmented_filenames)

                print(f"Read and augmented image: {image_file} - Size: {img.size}")
        except IOError:
            print(f"Error reading file: {image_file}")
    return images, filenames

def augmentFeatures(row, num_augmentations=5):
    augmented_rows = []
    for _ in range(num_augmentations):
        new_row = row.copy()
        for col in row.index:
            if col != 'Label' and isinstance(row[col], (int, float)):  
                noise = np.random.normal(0, 0.01)  
                new_row[col] += noise
        augmented_rows.append(new_row)
    return augmented_rows