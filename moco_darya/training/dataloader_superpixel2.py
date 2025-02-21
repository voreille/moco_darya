from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import random
import json
import pyspng

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform  

    def __call__(self, x):
        q = self.base_transform(x)  
        k = self.base_transform(x)  
        return q, k  

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR and MoCo v2"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma  

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])  
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))  

def get_moco_v2_augmentations():
    """
    Returns the MoCo v2 augmentation pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  
        transforms.RandomGrayscale(p=0.2),  
        GaussianBlur(sigma=[0.1, 2.0]),  # Use custom GaussianBlur
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    

class SuperpixelMoCoDataset(Dataset):
    def __init__(self, mapping_json, transform=None):
        with open(mapping_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def _load_image(self, path):
        """Loads an image efficiently using OpenCV."""
        # img = cv2.imread(path)  # OpenCV loads as BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        img = Image.fromarray(img).convert("RGB")  # Convert to PIL
        return img

    def __getitem__(self, idx):
        superpixel_data = self.superpixel_list[idx]
        tile_paths = superpixel_data["tile_paths"]

        tile_path_1 = random.choice(tile_paths)
        tile_path_2 = random.choice(tile_paths)

        image_1 = self._load_image(tile_path_1)
        image_2 = self._load_image(tile_path_2)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2

class SuperpixelMoCoDatasetNeighbor(Dataset):
    def __init__(self, mapping_json, transform=None):
        with open(mapping_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def _load_image(self, path):
        """Loads an image efficiently using OpenCV."""
        # img = cv2.imread(path)  # OpenCV loads as BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        img = Image.fromarray(img).convert("RGB")  # Convert to PIL
        return img

    def __getitem__(self, idx):
        superpixel_data = self.superpixel_list[idx]
        tile_paths = superpixel_data["tile_paths"]

        tile_path_1 = random.choice(tile_paths)
        tile_path_2 = random.choice(tile_paths)

        image_1 = self._load_image(tile_path_1)
        image_2 = self._load_image(tile_path_2)

        if self.transform:
            image_1_1 = self.transform(image_1)
            image_1_2 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1_1, image_1_2, image_2        


class SuperpixelMoCoDatasetNeighborAblation(Dataset):
    def __init__(self, mapping_json, transform=None):
        with open(mapping_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def _load_image(self, path):
        """Loads an image efficiently using OpenCV."""
        # img = cv2.imread(path)  # OpenCV loads as BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        with open(path, "rb") as f:
            img = pyspng.load(f.read())
        img = Image.fromarray(img).convert("RGB")  # Convert to PIL
        return img

    def __getitem__(self, idx):
        superpixel_data = self.superpixel_list[idx]
        tile_paths = superpixel_data["tile_paths"]

        tile_path_1 = random.choice(tile_paths)

        image_1 = self._load_image(tile_path_1)

        if self.transform:
            image_1_1 = self.transform(image_1)
            image_1_2 = self.transform(image_1)
            image_2 = self.transform(image_1)

        return image_1_1, image_1_2, image_2        


