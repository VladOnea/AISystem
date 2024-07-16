from PIL import Image
import os
import glob

def mirrorImage(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def loadImages(directory_path, pattern='*.bmp'):
    image_pattern = os.path.join(directory_path, pattern)
    images = []
    filenames = []

    for image_file in glob.glob(image_pattern):
        try:
            with Image.open(image_file) as img:
                images.append(img.copy())
                filenames.append(os.path.basename(image_file))  # Save the original filename
                mirrored_img = mirrorImage(img)
                images.append(mirrored_img)
                filenames.append("mirrored_" + os.path.basename(image_file))  # Save the mirrored filename
                print(f"Read image: {image_file} - Size: {img.size}")
        except IOError:
            print(f"Error reading file: {image_file}")
    return images, filenames
