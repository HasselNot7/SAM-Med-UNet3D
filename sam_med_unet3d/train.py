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

from sam_med_unet3d.sam_med_unet3d import SAMMedUNet3D

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
    sam_vit_cfg = dict(
        img_size=128,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_chans=256,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=0,
    )
    unet3d_cfg = dict(
        num_channels=1,
        feat_channels=[64, 128, 256, 512, 1024],
        residual='conv'
    )
    
    model = SAMMedUNet3D(sam_vit_cfg, unet3d_cfg, projector_out_channels=1024).to(device)
    
    # Load SAM weights
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sam_checkpoint', 'sam_med3d_turbo.pth')
    if os.path.exists(ckpt_path):
        print(f"Loading SAM weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith('image_encoder.'):
                new_k = k[len('image_encoder.'):]
                encoder_dict[new_k] = v
        
        if len(encoder_dict) > 0:
            msg = model.sam_encoder.load_state_dict(encoder_dict, strict=False)
            print(f"SAM Encoder loaded: {msg}")
        else:
            print("No image_encoder keys found in checkpoint!")
    else:
        print(f"Checkpoint not found at {ckpt_path}")

    # Freeze SAM Encoder
    for param in model.sam_encoder.parameters():
        param.requires_grad = False
    
    # Optimizer and Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
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
        if (epoch + 1) % 10 == 0:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f'unet3d_epfl_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    train()
