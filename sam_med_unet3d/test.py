import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tifffile
import numpy as np
from tqdm import tqdm
import argparse

# Add parent directory to path to import unet3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam_med_unet3d.sam_med_unet3d import SAMMedUNet3D
from sam_med_unet3d.train import EPFLDataset

def compute_metrics(pred, target):
    """
    Compute Dice and IoU metrics
    pred: (B, 1, D, H, W) - binary prediction (0 or 1)
    target: (B, 1, D, H, W) - binary target (0 or 1)
    """
    smooth = 1e-5
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    
    return dice.item(), iou.item()

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'epfl')
    
    # Dataset and DataLoader
    test_dataset = EPFLDataset(data_root, mode='test')
    if len(test_dataset) == 0:
        print("No test data found. Exiting.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Model Configuration
    sam_vit_cfg = dict(
        img_size=128,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_chans=384,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=0,
        use_rel_pos=True,
    )
    unet3d_cfg = dict(
        num_channels=1,
        feat_channels=[64, 128, 256, 512, 1024],
        residual='conv'
    )
    
    model = SAMMedUNet3D(sam_vit_cfg, unet3d_cfg, projector_out_channels=1024).to(device)
    
    # Manually initialize projector to allow loading weights
    # Based on training logic: vit_feat = (B, C, D, H, W) -> permute(0, 4, 1, 2, 3) -> (B, W, C, D, H)
    # So in_channels = W = img_size // patch_size
    projector_in_channels = sam_vit_cfg['img_size'] // sam_vit_cfg['patch_size']
    from sam_med_unet3d.sam_med_unet3d import Conv3DProjector
    model.projector = Conv3DProjector(projector_in_channels, 1024).to(device)
    
    # Load trained weights
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        
        # Handle relative position embedding size mismatch if loading from original SAM checkpoint
        # But here we are loading our own trained checkpoint, so shapes should match.
        # However, if we are loading the pretrained SAM weights directly for testing (zero-shot), we might need resizing.
        # Assuming we load a fine-tuned checkpoint where shapes are already correct.
        
        # If loading fails due to size mismatch, try resizing (similar to train.py)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if 'size mismatch' in str(e) and 'rel_pos' in str(e):
                print("Size mismatch detected in relative position embeddings. Attempting to resize...")
                model_state = model.state_dict()
                for k, v in state_dict.items():
                    if k in model_state and v.shape != model_state[k].shape and 'rel_pos' in k:
                        print(f"Resizing {k}: {v.shape} -> {model_state[k].shape}")
                        v = v.permute(1, 0).unsqueeze(0)
                        v = F.interpolate(v, size=model_state[k].shape[0], mode='linear', align_corners=False)
                        v = v.squeeze(0).permute(1, 0)
                        state_dict[k] = v
                model.load_state_dict(state_dict)
            else:
                raise e
    else:
        print(f"Checkpoint not found at {args.checkpoint}")
        return

    model.eval()
    
    total_dice = 0
    total_iou = 0
    num_samples = 0
    
    print(f"Start testing on {device}...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            pred_mask = (outputs > 0.5).float()
            
            dice, iou = compute_metrics(pred_mask, labels)
            
            total_dice += dice
            total_iou += iou
            num_samples += 1
            
            pbar.set_postfix({'dice': dice, 'iou': iou})
            
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    
    print("\n" + "="*30)
    print(f"Test Results:")
    print(f"Number of samples: {num_samples}")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    args = parser.parse_args()
    
    test(args)
