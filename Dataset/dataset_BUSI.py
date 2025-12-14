import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BUSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(").png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        # Correct mask name
        mask_name = img_name.replace(".png", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarize mask
        mask = (mask > 0).astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]          # (1,H,W)
            mask = augmented["mask"]            # (H,W)

        mask = mask.long()
        return image, mask
