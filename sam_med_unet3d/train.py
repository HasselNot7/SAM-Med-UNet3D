import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import unet3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet3d.unet3d import UNet3D

class EPFLDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.image_dir = os.path.join(root_dir, mode, 'images')
        self.label_dir = os.path.join(root_dir, mode, 'labels')
        
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, '*.tif')))
        
        # Check if files exist
        if len(self.image_files) == 0:
            print(f"Warning: No images found in {self.image_dir}")
        if len(self.label_files) == 0:
            print(f"Warning: No labels found in {self.label_dir}")
            
        assert len(self.image_files) == len(self.label_files), f"Number of images ({len(self.image_files)}) and labels ({len(self.label_files)}) must match"
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        img = tifffile.imread(img_path)
        label = tifffile.imread(label_path)
        
        # Normalize image to 0-1
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img - img.min()
        
        # Ensure label is 0/1
        label = (label > 0).astype(np.float32)
        
        # Add channel dimension (C, D, H, W)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        label = np.expand_dims(label, axis=0).astype(np.float32)
        
        return torch.from_numpy(img), torch.from_numpy(label)

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def train():
    # Configuration
    # Assuming the script is run from sam_med_unet3d/ or root, we need absolute path or correct relative path
    # Based on workspace info: /home/hasselnot/ML/SAM-Med3D/datasets/epfl
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'epfl')
    
    batch_size = 1 # 3D data consumes a lot of memory
    lr = 1e-4
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Data root: {data_root}")
    
    # Dataset and DataLoader
    train_dataset = EPFLDataset(data_root, mode='train')
    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Model
    # num_channels=1 because input is grayscale and output is binary mask (sigmoid)
    model = UNet3D(num_channels=1, residual='conv').to(device)
    
    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    print(f"Start training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate Dice score for monitoring
            pred_mask = (outputs > 0.5).float()
            dice = dice_coeff(pred_mask, labels)
            epoch_dice += dice.item()
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")
        
        # Save checkpoint
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f'unet3d_epfl_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    train()
