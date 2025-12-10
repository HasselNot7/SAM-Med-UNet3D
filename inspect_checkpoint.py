import torch
import os
import sys

ckpt_path = 'sam_checkpoint/sam_med3d_turbo.pth'
if not os.path.exists(ckpt_path):
    # Try absolute path based on previous context if relative fails
    ckpt_path = '/home/hasselnot/ML/SAM-Med3D/sam_checkpoint/sam_med3d_turbo.pth'

if os.path.exists(ckpt_path):
    print(f"Inspecting checkpoint: {ckpt_path}")
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        print(f"Top level keys: {list(state_dict.keys())}")
        
        if 'model' in state_dict:
            print("Found 'model' key. Inspecting its content...")
            model_dict = state_dict['model']
            keys = list(model_dict.keys())
            print(f"Total keys in model_dict: {len(keys)}")
            print("First 20 keys in model_dict:")
            for k in keys[:20]:
                print(k)
                
            # Check for image_encoder
            img_enc_keys = [k for k in keys if 'image_encoder' in k]
            print(f"\nKeys containing 'image_encoder': {len(img_enc_keys)}")
            if len(img_enc_keys) > 0:
                print("Example keys:")
                for k in img_enc_keys[:5]:
                    print(k)
        else:
            print("'model' key not found. Inspecting top level keys...")
            keys = list(state_dict.keys())
            print(f"Total keys: {len(keys)}")
            print("First 20 keys:")
            for k in keys[:20]:
                print(k)
                
            img_enc_keys = [k for k in keys if 'image_encoder' in k]
            print(f"\nKeys containing 'image_encoder': {len(img_enc_keys)}")
            if len(img_enc_keys) > 0:
                print("Example keys:")
                for k in img_enc_keys[:5]:
                    print(k)

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print(f"Checkpoint not found at {ckpt_path}")
