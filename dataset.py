import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Random circle
        radius = np.random.randint(5, 15)
        x_center = np.random.randint(radius, self.img_size - radius)
        y_center = np.random.randint(radius, self.img_size - radius)

        y, x = np.ogrid[:self.img_size, :self.img_size]
        circle = (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2

        img[circle] = 1.0
        mask[circle] = 1.0

        img = torch.tensor(img).unsqueeze(0)   # (1, H, W)
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask
    
    
import matplotlib.pyplot as plt
from dataset import ShapeDataset

# Create dataset
dataset = ShapeDataset(num_samples=5, img_size=64)

# Get one sample
img, mask = dataset[0]

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)

# Visualize
plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(img.squeeze(), cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask.squeeze(), cmap="gray")

plt.show()



