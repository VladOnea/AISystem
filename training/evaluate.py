import torch
from models.unet import UNet
import matplotlib.pyplot as plt
from training.visualization import visualizePredictions, visualizeHeatmap, skeletonizedImages, visualizeDecisionTree
from skimage.morphology import skeletonize
from featureCalculation.feature import calculateVesselMetrics, calculateTortuosity
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def evaluateModel(val_loader, test_loader, criterion, criterion2, pixel_spacing_mm, output_excel_path, label_excel_path):
    unet = UNet(n_channels=1, n_classes=1)
    unet.load_state_dict(torch.load('unet_model_best_vein.pth'))
    unet.eval()

    # Load label information
    labels_df = pd.read_excel(label_excel_path)
    labels_df['Label'] = labels_df['Disease'].apply(lambda x: 1 if x == 'DR' else 0)
    labels_df['ID'] = labels_df['ID'].astype(str).str.strip() 

    total_val_loss = 0
    features = []
    
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data[:2]
            val_outputs = unet(val_inputs)
            val_loss = criterion(val_outputs, val_labels) * 0.4 + criterion2(val_outputs, val_labels) * 0.6
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    print(f'Average validation loss: {average_val_loss}')

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            if i >= 3:
                break
            val_inputs, val_labels = val_data[:2]
            val_outputs = unet(val_inputs)
            visualizePredictions(val_inputs.cpu(), val_labels.cpu(), val_outputs.cpu(), 0)

    unet.eval()
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs, test_labels = test_data[:2]
            test_filenames = test_data[2] if len(test_data) > 2 else None
            test_outputs = unet(test_inputs)
            preds = torch.sigmoid(test_outputs) > 0.5
            for i in range(test_inputs.size(0)):
                correct_pixels = (preds[i] == test_labels[i]).sum().item()
                total_pixels = torch.numel(preds[i])
                image_accuracy = correct_pixels / total_pixels * 100
                if test_filenames is not None:
                    print(f'Image {test_filenames[i]} has accuracy: {image_accuracy:.2f}%')

    torch.save(unet.state_dict(), 'unet_model_final_test_evaluated_vein.pth')

    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data[0]
            outputs = unet(test_inputs)
            preds = torch.sigmoid(outputs)
            input_image_np = test_inputs[0].squeeze().cpu().numpy()
            prediction_np = preds[0].squeeze().cpu().numpy()
            visualizeHeatmap(input_image_np, prediction_np)
            break

    with torch.no_grad():
        for test_data in test_loader:
            test_inputs, test_labels = test_data[:2]
            test_filenames = test_data[2] if len(test_data) > 2 else None
            test_outputs = unet(test_inputs)
            preds = torch.sigmoid(test_outputs) > 0.5
            for j in range(test_inputs.size(0)):
                prediction_np = preds[j].squeeze().cpu().numpy()
                skeleton = skeletonize(prediction_np)
                density, vessel_length_mm = calculateVesselMetrics(prediction_np, pixel_spacing_mm)
                tortuosity = calculateTortuosity(skeleton)

                if test_filenames is not None:
                    file_id = os.path.splitext(test_filenames[j])[0]  # Remove file extension
                    file_id = file_id.strip()  # Normalize filename to match IDs in labels_df
                    if file_id not in labels_df['ID'].values:
                        print(f'Label for {file_id} not found in labels_df.')
                    else:
                        label_row = labels_df[labels_df['ID'] == file_id]
                        label = label_row['Label'].values[0]
                        features.append([file_id, density, tortuosity, label])
                        skeletonizedImages(prediction_np, skeleton)
                        print(f'Image {file_id}: Vessel Density: {density:.4f}, Vessel Length: {vessel_length_mm:.2f} mm, Tortuosity: {tortuosity:.4f}')
                        print(f'Appending to features: ID={file_id}, Density={density}, Tortuosity={tortuosity}, Label={label}')

    features_df = pd.DataFrame(features, columns=['ID', 'Density', 'Tortuosity', 'Label'])
    features_df.to_excel(output_excel_path, index=False)

    return features_df


def trainDecisionTree(features_df):
    X = features_df[['Density', 'Tortuosity']]
    y = features_df['Label']
    
    y = y.astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train decision tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    visualizeDecisionTree(clf, feature_names=['Density', 'Tortuosity'], class_names=['No DR', 'DR'])
    
    return clf

