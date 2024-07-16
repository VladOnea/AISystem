import os
import sys
from data.dataset import ImageDataset
from data.utils import loadImages
from data.transforms import transform
from training.train import trainModel
from training.evaluate import evaluateModel, trainDecisionTree
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from training.visualization import visualizeInputs
import torch.nn as nn
from models.diceLoss import DiceLoss
import subprocess

def main(image_dir, label_dir):
    images, image_filenames = loadImages(image_dir)
    labels, labels_filenames= loadImages(label_dir)

    print("Total number of BMP images read:", len(images))

    # Visualize images and labels
    visualizeInputs(images, labels)
    
    # Split the dataset into training/validation and test sets
    train_val_images, test_images, train_val_labels, test_labels, train_val_filenames, test_filenames = train_test_split(
        images, labels, image_filenames, test_size=0.2, random_state=42)
    
    # Split the training/validation set into training and validation sets
    train_images, val_images, train_labels, val_labels, train_filenames, val_filenames = train_test_split(
        train_val_images, train_val_labels, train_val_filenames, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    # Create datasets
    train_dataset = ImageDataset(train_images, train_labels, train_filenames, transform=transform)
    val_dataset = ImageDataset(val_images, val_labels, val_filenames, transform=transform)
    test_dataset = ImageDataset(test_images, test_labels, test_filenames, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    pixel_spacing_mm = 3 / 304
    
    # Train the model
    unet = trainModel(train_loader, val_loader, num_epochs=5)
    
    features_excel_path = 'Image_Features.xlsx'
    label_excel_path = 'Updated_Text_labels.xlsx'
    # Evaluate the model
    features_df = evaluateModel(val_loader, test_loader, criterion, criterion2, pixel_spacing_mm, features_excel_path, label_excel_path)

    # Train and evaluate decision tree
    clf = trainDecisionTree(features_df)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <image_dir> <label_dir>")
        sys.exit(1)
    image_dir = sys.argv[1]
    label_dir = sys.argv[2]
    main(image_dir, label_dir)

    subprocess.run(["python", "updateExcelLabels.py"])
