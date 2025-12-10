import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile
from tqdm import tqdm

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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class EPFLDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, 'images', '*.tif')))
        self.label_paths = sorted(
            glob.glob(os.path.join(root_dir, 'labels', '*.tif')))

        if len(self.image_paths) == 0:
            print(
                f"Warning: No images found in {os.path.join(root_dir, 'images')}")

        if len(self.image_paths) != len(self.label_paths):
            print(
                f"Warning: Mismatch in image ({len(self.image_paths)}) and label ({len(self.label_paths)}) count")
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


def train(batch_size=1, learning_rate=1e-4, num_epochs=16):
    # Configuration
    # Assuming the script is in unet3d/train_test.py and data is in datasets/epfl/train
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, 'datasets', 'epfl', 'train')

    batch_size = batch_size  # Reduced batch size for safety
    learning_rate = learning_rate
    num_epochs = num_epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Data root: {data_root}")

    # Dataset and DataLoader
    dataset = EPFLDataset(data_root)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # num_workers=0 for compatibility
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    print(f"Dataset size: {len(dataset)}")

    # Model
    # Using smaller feature channels to ensure it fits in memory and runs faster
    model = UNet3D(num_channels=1, feat_channels=[
                   16, 32, 64, 128, 256]).to(device)

    # Loss and Optimizer
    bce_criterion = nn.BCELoss()
    dice_criterion = DiceLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning Rate Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training Loop
    total_pbar = tqdm(range(num_epochs), desc="Total Training")
    for epoch in total_pbar:
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss_bce = bce_criterion(outputs, labels)
            loss_dice = dice_criterion(outputs, labels)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}", bce=f"{loss_bce.item():.4f}", dice=f"{loss_dice.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        
        scheduler.step()
        
        total_pbar.set_postfix(avg_loss=f"{epoch_loss:.4f}")
        tqdm.write(
            f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, f'unet3d_epfl_epoch_{epoch+1}.pth'))
            tqdm.write(f"Saved checkpoint to {checkpoint_dir}")


if __name__ == '__main__':
    train(batch_size=2, learning_rate=1e-4, num_epochs=50)
