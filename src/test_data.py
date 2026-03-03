import torch
from torchvision import datasets, transforms
import os

DATA_DIR = r"C:\Users\Nishant\Downloads\NDxGenius\PlantPal\data\plantvillage"
# If you find a subfolder, update the path above accordingly.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes.")
    print("First few classes:", dataset.classes[:5])
    
    # Try to load one batch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Attempting to load first image...")
    for images, labels in loader:
        print(f"Success! Image shape: {images.shape}")
        break
    else:
        print("No images found.")
except Exception as e:
    print(f"Error: {e}")