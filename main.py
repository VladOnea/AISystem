import os
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
from data.updateExcelLabels import updateLabelsExcel
from classificators.classificator import compareModels

def main():
    image_dir = 'C:\\Users\\bolog\\OneDrive\\Desktop\\LicentaVlad\\Images'
    label_dir = 'C:\\Users\\bolog\\OneDrive\\Desktop\\LicentaVlad\\LabelsVein'

    images, image_filenames = loadImages(image_dir)
    labels, label_filenames = loadImages(label_dir)

    print("Total number of BMP images read:", len(images))

    visualizeInputs(images, labels)
    
    file_path = 'C:\\Users\\bolog\\OneDrive\\Desktop\\LicentaVlad\\11.07\\Text labels.xlsx'
    updated_file_path = 'C:\\Users\\bolog\\OneDrive\\Desktop\\LicentaVlad\\11.07\\Updated_Text_labels.xlsx'
    updateLabelsExcel(file_path,updated_file_path)
    
    train_val_images, test_images, train_val_labels, test_labels, train_val_filenames, test_filenames = train_test_split(
        images, labels, image_filenames, test_size=0.2, random_state=42)
    
    train_images, val_images, train_labels, val_labels, train_filenames, val_filenames = train_test_split(
        train_val_images, train_val_labels, train_val_filenames, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    train_dataset = ImageDataset(train_images, train_labels, train_filenames, transform=transform)
    val_dataset = ImageDataset(val_images, val_labels, val_filenames, transform=transform)
    test_dataset = ImageDataset(test_images, test_labels, test_filenames, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    pixel_spacing_mm = 3 / 304
    
    unet = trainModel(train_loader, val_loader, num_epochs=5)
    
    features_excel_path = 'Image_Features.xlsx'
    label_excel_path = 'Updated_Text_labels.xlsx'

    features_df = evaluateModel(val_loader, test_loader, criterion, criterion2, pixel_spacing_mm,features_excel_path,label_excel_path)

    compareModels(features_excel_path)

if __name__ == "__main__":
    main()
