import torch
import torch.optim as optim
from models.unet import UNet
from models.diceLoss import DiceLoss
from training.visualization import visualizeLoss

def trainModel(train_loader, val_loader, num_epochs=5, learning_rate=0.001, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(n_channels=1, n_classes=1)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_limit = 3  
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        unet.train()
        total_train_loss = 0

        for data in train_loader:
            inputs, labels, _ = data
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = unet(inputs)
            bce_loss = criterion(outputs, labels)
            dice_loss = criterion2(outputs, labels)
            loss = 0.4 * bce_loss + 0.6 * dice_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        training_losses.append(average_train_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {average_train_loss}')

        # Validation loop
        unet.eval()
        total_val_loss = 0

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels, _ = val_data
                val_inputs = val_inputs.to(device=device, dtype=torch.float32)
                val_labels = val_labels.to(device=device, dtype=torch.long)
                val_outputs = unet(val_inputs)
                val_loss = criterion(val_outputs, val_labels) * 0.4 + criterion2(val_outputs, val_labels) * 0.6
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(average_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {average_val_loss}')

        # Model Checkpointing
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            try:
                torch.save(unet.state_dict(), 'unet_model_best_vein.pth')
                early_stopping_counter = 0  
            except Exception as e:
                print(f"Error saving the model: {e}")
        else:
            early_stopping_counter += 1

        # Early Stopping
        if early_stopping_counter > early_stopping_limit:
            print("Early stopping triggered")
            break

    # Visualize training and validation losses
    visualizeLoss(training_losses, validation_losses, num_epochs)
    
    try:
        torch.save(unet.state_dict(), 'unet_model_final_vein.pth')
    except Exception as e:
        print(f"Error saving the model: {e}")
    
    return unet
