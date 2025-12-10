import os
import sys
import glob
import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader

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

def compute_dice(pred, target):
    """
    Compute Dice coefficient for a single sample or batch.
    pred: (B, C, D, H, W) or (C, D, H, W) - binary (0 or 1)
    target: (B, C, D, H, W) or (C, D, H, W) - binary (0 or 1)
    """
    smooth = 1e-5
    
    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def compute_iou(pred, target):
    """
    Compute IoU for a single sample or batch.
    """
    smooth = 1e-5
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)

def test():
    # Configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, 'datasets', 'epfl', 'test')
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    
    # Find latest checkpoint
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    checkpoint_path = checkpoints[-1] # Use the last one (likely the latest epoch)
    print(f"Testing with checkpoint: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = EPFLDataset(data_root)
    if len(dataset) == 0:
        print("Test dataset is empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test dataset size: {len(dataset)}")

    # Model
    model = UNet3D(num_channels=1, feat_channels=[16, 32, 64, 128, 256]).to(device)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.numpy() # Keep labels on CPU for metric calc

            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy()

            dice = compute_dice(preds, labels)
            iou = compute_iou(preds, labels)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            
            if (i + 1) % 10 == 0:
                print(f"Sample {i+1}/{len(dataset)} - Dice: {dice:.4f}, IoU: {iou:.4f}")

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print("\n" + "="*30)
    print(f"Test Results:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score:  {avg_iou:.4f}")
    print("="*30)

if __name__ == '__main__':
    test()
