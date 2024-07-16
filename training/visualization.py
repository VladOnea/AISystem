import matplotlib.pyplot as plt
import torch
from sklearn.tree import plot_tree

def visualizeLoss(training_losses, validation_losses, num_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def showImageLabel(image, label, title="Image and Label"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray', interpolation='nearest')  # Display the original image
    plt.imshow(label, cmap='jet', alpha=0.5, interpolation='nearest') 
    plt.title('Label')
    plt.axis('off')
    plt.suptitle(title)
    plt.show()


def visualizeInputs(images, images_labels): #function for visualizing 4 input images
    for i in range(0, 400, 100):  
        showImageLabel(images[i], images_labels[i], title=f"Original Image and Label {i}")

def visualizeHeatmap(input_image, prediction, title="Prediction Heatmap"):
    plt.figure(figsize=(12, 6))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Display the heatmap
    plt.subplot(1, 2, 2)
    # Assuming prediction is a 2D tensor (H, W) with probability values
    plt.imshow(input_image, cmap='gray', interpolation='nearest')  # Display the original image
    plt.imshow(prediction, cmap='jet', alpha=0.5, interpolation='nearest')  # Overlay the heatmap
    plt.title(title)
    plt.axis('off')

    plt.show()

def visualizePredictions(inputs, labels, preds, index):
    plt.figure(figsize=(12, 6))

    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(inputs[index][0], cmap='gray')  # Inputs is [batch, channel, H, W]
    plt.title('Input Image')
    plt.axis('off')

    # True Label
    plt.subplot(1, 3, 2)
    label = torch.squeeze(labels[index])  # Remove the channel dimension
    plt.imshow(label, cmap='gray')  # Now label is [H, W]
    plt.title('True Label')
    plt.axis('off')

    # Predicted Label
    plt.subplot(1, 3, 3)
    preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
    preds_binary = (preds[index] > 0.5).float()  # Threshold probabilities to get binary predictions
    pred = torch.squeeze(preds_binary[0])  # Remove the channel dimension
    plt.imshow(pred, cmap='gray')  # [batch, channel, H, W]
    plt.title('Predicted Label')
    plt.axis('off')

    plt.show()

def skeletonizedImages(prediction_np,skeleton):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(prediction_np, cmap='gray')
    plt.title('Binary Segmentation')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Skeletonized Vessels')
    plt.axis('off')
    plt.show()

def visualizeDecisionTree(clf, feature_names, class_names, output_path='decision_tree.png'):
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
    plt.savefig(output_path)
    plt.show()