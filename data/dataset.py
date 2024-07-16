from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, images, labels, filenames, transform=None):
        self.images = images
        self.labels = labels
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        if len(label.shape) == 2:
            label = torch.unsqueeze(label, 0)

        return image, label, filename