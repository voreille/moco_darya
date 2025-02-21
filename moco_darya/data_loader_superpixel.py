from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import random
import json

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
    """
    A dataset that draws two random tiles from the same superpixel and applies MoCo transformations.

    Args:
        mapping_json (str): Path to the JSON file containing the mapping.
        transform (callable, optional): MoCo-style augmentation transform.
    """

    def __init__(self, mapping_json, transform=None):
        # Load the mapping as a list of dictionaries
        with open(mapping_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def __getitem__(self, idx):
        """
        Returns two different tiles belonging to the same superpixel.

        Returns:
            (Tensor, Tensor): Two augmented images.
        """
        superpixel_data = self.superpixel_list[idx]
        tile_paths = superpixel_data["tile_paths"]

        # Sample two tiles (with replacement)
        tile_path_1 = random.choice(tile_paths)
        tile_path_2 = random.choice(tile_paths)

        # Load images
        image_1 = Image.open(tile_path_1).convert("RGB")
        image_2 = Image.open(tile_path_2).convert("RGB")

        # Apply MoCo transformations
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2


class SuperpixelMoCoDatasetDebug(Dataset):
    """
    A dataset that draws a configurable number of random tiles from the same superpixel 
    and applies MoCo transformations.

    Args:
        mapping_json (str): Path to the JSON file containing the mapping.
        num_tiles (int): Number of tiles to return per sample.
        transform (callable, optional): MoCo-style augmentation transform.
    """

    def __init__(self, mapping_json, num_tiles=2, transform=None):
        assert num_tiles > 0, "num_tiles must be at least 1"

        # Load the mapping as a list of dictionaries
        with open(mapping_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.num_tiles = num_tiles
        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def __getitem__(self, idx):
        """
        Returns a specified number of tiles belonging to the same superpixel.

        Returns:
            List of transformed images.
        """
        superpixel_data = self.superpixel_list[idx]
        tile_paths = superpixel_data["tile_paths"]
        nb_tiles = len(tile_paths)

        # Sample `num_tiles` tiles (with replacement)
        sampled_tile_paths = random.choices(tile_paths, k=self.num_tiles)

        # Load images and apply transformations
        images = [
            Image.open(path).convert("RGB") for path in sampled_tile_paths
        ]

        if self.transform:
            images = [self.transform(img) for img in images]

        tile_ids = [Path(p).stem for p in sampled_tile_paths]

        return images, tile_ids, nb_tiles