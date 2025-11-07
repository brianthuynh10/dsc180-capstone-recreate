from PIL import Image
import torch
import numpy as np


class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)

        # Scale to [0,1] if needed
        if image.max() > 1:
            image /= image.max()

        pil_image = Image.fromarray((image * 255).astype(np.uint8)).convert("L")

        # Apply transform (should output 1×224×224 tensor)
        if self.transform:
            pil_image = self.transform(pil_image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32).view(-1)
        return pil_image, label

