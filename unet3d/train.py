import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile

# Add the current directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

try:
    from unet3d import UNet3D
except ImportError:
    try:
        from unet3d.unet3d import UNet3D
    except ImportError:
        # Fallback if running from within the folder
        from unet3d import UNet3D

class EPFLDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'images', '*.tif')))
        self.label_paths = sorted(glob.glob(os.path.join(root_dir, 'labels', '*.tif')))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {os.path.join(root_dir, 'images')}")
            
        if len(self.image_paths) != len(self.label_paths):
            print(f"Warning: Mismatch in image ({len(self.image_paths)}) and label ({len(self.label_paths)}) count")
            # Take the intersection or minimum
            min_len = min(len(self.image_paths), len(self.label_paths))
            self.image_paths = self.image_paths[:min_len]
            self.label_paths = self.label_paths[:min_len]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        img = tifffile.imread(img_path)
        label = tifffile.imread(label_path)

        # Normalize image to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Process label
        label = label.astype(np.float32)
        if label.max() > 1:
            label = label / 255.0
        label = (label > 0.5).astype(np.float32)

        # Add channel dimension (C, D, H, W) -> (1, 128, 128, 128)
        img = np.expand_dims(img, axis=0)
        label = np.expand_dims(label, axis=0)

        return torch.from_numpy(img), torch.from_numpy(label)

def train():
    # Configuration
    # Assuming the script is in unet3d/train_test.py and data is in datasets/epfl/train
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, 'datasets', 'epfl', 'train')
    
    batch_size = 1 # Reduced batch size for safety
    learning_rate = 1e-4
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Data root: {data_root}")

    # Dataset and DataLoader
    dataset = EPFLDataset(data_root)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for compatibility
    
    print(f"Dataset size: {len(dataset)}")

    # Model
    # Using smaller feature channels to ensure it fits in memory and runs faster
    model = UNet3D(num_channels=1, feat_channels=[16, 32, 64, 128, 256]).to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'unet3d_epfl_epoch_{epoch+1}.pth'))
        print(f"Saved checkpoint to {checkpoint_dir}")

if __name__ == '__main__':
    train()
