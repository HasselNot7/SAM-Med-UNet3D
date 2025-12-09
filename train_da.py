import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from monai.losses import DiceCELoss

# 导入自定义模块
from sam_med_unet3d.sam_med_unet3d import SAMMedUNet3D
from unet3d.unet3d import UNet3D
try:
    from dataset_hdf5 import HDF5Dataset
except ImportError:
    print("Warning: dataset_hdf5 module not found or h5py not installed.")
    HDF5Dataset = None

# -----------------------------------------------------------------------------
# 1. 损失函数定义
# -----------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    感知损失 = 特征重构损失 + 风格重构损失
    """
    def __init__(self):
        super().__init__()

    def forward(self, s_feat, t_feat):
        """
        s_feat: 源域特征 (B, C, D, H, W)
        t_feat: 目标域特征 (B, C, D, H, W)
        """
        # 1. 特征重构损失 (L_feat)
        # 计算欧氏距离的平方，并归一化
        diff = s_feat - t_feat
        l_feat = torch.mean(diff ** 2)  # Mean相当于除以CHW

        # 2. 风格重构损失 (L_style)
        # 计算Gram矩阵
        def gram_matrix(feat):
            B, C, D, H, W = feat.shape
            feat = feat.view(B, C, -1)  # (B, C, D*H*W)
            # Gram矩阵: (B, C, C)
            gram = torch.bmm(feat, feat.transpose(1, 2))
            return gram / (C * D * H * W)

        gram_s = gram_matrix(s_feat)
        gram_t = gram_matrix(t_feat)
        l_style = torch.mean((gram_s - gram_t) ** 2)

        return l_feat + l_style

class MinEntLoss(nn.Module):
    """熵最小化损失"""
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        """
        logits: (B, C, D, H, W)
        """
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.mean()

def load_sam_encoder_weights(model, checkpoint_path):
    """
    从 SAM-Med3D 完整权重中加载 image_encoder 部分
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    print(f"Loading SAM encoder weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('image_encoder.'):
            new_key = k[len('image_encoder.'):]
            encoder_state_dict[new_key] = v
    
    if len(encoder_state_dict) == 0:
        print("No image_encoder weights found in checkpoint! Trying to load as pure encoder...")
        # 尝试直接加载（假设checkpoint就是encoder）
        msg = model.sam_encoder.load_state_dict(state_dict, strict=False)
    else:
        msg = model.sam_encoder.load_state_dict(encoder_state_dict, strict=True)
        
    print(f"Loaded SAM encoder weights: {msg}")

# -----------------------------------------------------------------------------
# 2. 训练流程
# -----------------------------------------------------------------------------

def train_domain_adaptation(
    model: SAMMedUNet3D,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    num_epochs: int = 100,
    lr: float = 1e-4,
    save_dir: str = 'checkpoints'
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=lr)
    
    # 损失函数
    dice_ce_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    perceptual_loss = PerceptualLoss()
    min_ent_loss = MinEntLoss()

    # 阶段控制
    # 假设前30% epoch为阶段1，中间30%为阶段2，最后40%为阶段3
    phase1_epochs = int(num_epochs * 0.3)
    phase2_epochs = int(num_epochs * 0.3)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # 确定当前阶段
        if epoch < phase1_epochs:
            phase = 1
            print(f"Epoch {epoch+1}/{num_epochs} - Phase 1: Source Domain Supervised Training")
        elif epoch < phase1_epochs + phase2_epochs:
            phase = 2
            print(f"Epoch {epoch+1}/{num_epochs} - Phase 2: Unsupervised Domain Adaptation")
            # 冻结UNet部分，只训练SAM编码器和投影器
            for param in model.unet3d.parameters():
                param.requires_grad = False
            for param in model.sam_encoder.parameters():
                param.requires_grad = True
            if model.projector:
                for param in model.projector.parameters():
                    param.requires_grad = True
        else:
            phase = 3
            print(f"Epoch {epoch+1}/{num_epochs} - Phase 3: Target Domain Self-Training")
            # 解冻UNet，继续训练所有参数
            for param in model.parameters():
                param.requires_grad = True

        # 迭代器
        target_iter = iter(target_loader)
        
        pbar = tqdm(source_loader)
        for batch_idx, (s_img, s_label) in enumerate(pbar):
            s_img = s_img.to(device)
            s_label = s_label.to(device)
            
            # 获取目标域数据
            try:
                t_img, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                t_img, _ = next(target_iter)
            t_img = t_img.to(device)

            optimizer.zero_grad()
            loss = 0

            # -------------------------------------------------------
            # Phase 1: 源域监督训练
            # -------------------------------------------------------
            if phase == 1:
                # 前向传播
                s_pred = model(s_img)
                # 计算Dice+CE损失
                loss = dice_ce_loss(s_pred, s_label)

            # -------------------------------------------------------
            # Phase 2: 无监督域适应 (Perceptual Loss)
            # -------------------------------------------------------
            elif phase == 2:
                # 获取SAM编码器特征
                s_feat = model.sam_encoder(s_img)
                t_feat = model.sam_encoder(t_img)
                
                # 计算感知损失
                loss = perceptual_loss(s_feat, t_feat)

            # -------------------------------------------------------
            # Phase 3: 目标域自训练与不确定性精炼
            # -------------------------------------------------------
            elif phase == 3:
                # 1. 源域监督损失 (保持基本性能)
                s_pred = model(s_img)
                loss_sup = dice_ce_loss(s_pred, s_label)
                
                # 2. 目标域伪标签训练
                # 生成伪标签
                with torch.no_grad():
                    t_pred_raw = model(t_img)
                    t_prob = torch.sigmoid(t_pred_raw)
                    
                    # 计算不确定性 (Entropy)
                    # 二分类熵: -p*log(p) - (1-p)*log(1-p)
                    entropy = -(t_prob * torch.log(t_prob + 1e-8) + 
                              (1 - t_prob) * torch.log(1 - t_prob + 1e-8))
                    
                    # 生成硬伪标签
                    t_pseudo_label = (t_prob > 0.5).float()
                
                # 动态加权: 1 - Entropy (归一化)
                weight = 1 - (entropy / entropy.max())
                weight = weight.detach()
                
                # 伪标签损失 (加权CE)
                # 这里简化为加权MSE或自定义CE，因为DiceCELoss不好直接加权像素级
                # 使用加权BCE
                bce_loss = F.binary_cross_entropy_with_logits(t_pred_raw, t_pseudo_label, reduction='none')
                loss_pseudo = (weight * bce_loss).mean()
                
                # 3. 熵最小化损失
                loss_min_ent = min_ent_loss(t_pred_raw)
                
                # 总损失
                lambda_target = 0.1
                lambda_min_ent = 0.1
                loss = loss_sup + lambda_target * loss_pseudo + lambda_min_ent * loss_min_ent

            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")

        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    # 配置参数
    # 注意：SAM-Med3D 预训练模型通常是在 128x128x128 的输入上训练的
    # 如果你的 crop_size 是 (32, 32, 32)，你需要调整 img_size 参数或者对 pos_embed 进行插值
    # 这里我们将 img_size 设置为 32 以匹配 crop_size，但这会导致 pos_embed 形状不匹配
    # 更好的做法是保持 img_size=128，并在 dataset 中 resize 到 128，或者在模型中处理 pos_embed 插值
    
    # 方案 A: 调整 crop_size 为 128 (显存可能不够)
    # 方案 B: 调整 img_size 为 32，并允许加载权重时 pos_embed 不匹配 (需要手动插值)
    
    # 这里我们采用方案 B 的变体：在加载权重后，对 pos_embed 进行插值以适应当前的 img_size (32)
    
    current_img_size = 32  # 匹配 dataset 的 crop_size
    
    sam_vit_cfg = dict(
        img_size=current_img_size,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    unet3d_cfg = dict(
        num_channels=1,
        feat_channels=[64, 128, 256, 512, 1024],
        residual='conv'
    )
    
    # 初始化模型
    model = SAMMedUNet3D(sam_vit_cfg, unet3d_cfg, projector_out_channels=1024)
    
    # 加载预训练权重 (如果有)
    sam_checkpoint = "/home/hasselnot/ML/SAM-Med3D/sam_checkpoint/sam_med3d_turbo.pth"
    
    # 自定义加载逻辑，处理 pos_embed 尺寸不匹配问题
    if os.path.exists(sam_checkpoint):
        print(f"Loading SAM encoder weights from {sam_checkpoint}...")
        state_dict = torch.load(sam_checkpoint, map_location='cpu')
        
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('image_encoder.'):
                new_key = k[len('image_encoder.'):]
                encoder_state_dict[new_key] = v
        
        # 处理 pos_embed 插值
        if 'pos_embed' in encoder_state_dict:
            pos_embed = encoder_state_dict['pos_embed'] # (1, D, H, W, C) or similar
            # 假设 pos_embed 是 (1, 8, 8, 8, 768) 对应 128输入/16patch
            # 我们现在需要 (1, 2, 2, 2, 768) 对应 32输入/16patch
            
            target_shape = model.sam_encoder.pos_embed.shape
            if pos_embed.shape != target_shape:
                print(f"Resizing pos_embed from {pos_embed.shape} to {target_shape}")
                # Permute to (1, C, D, H, W) for interpolate
                pos_embed = pos_embed.permute(0, 4, 1, 2, 3)
                pos_embed = F.interpolate(pos_embed, size=target_shape[1:4], mode='trilinear', align_corners=False)
                # Permute back
                pos_embed = pos_embed.permute(0, 2, 3, 4, 1)
                encoder_state_dict['pos_embed'] = pos_embed
        
        msg = model.sam_encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded SAM encoder weights: {msg}")
    
    # -------------------------------------------------------------------------
    # 真实数据加载示例 (取消注释并填入路径)
    # -------------------------------------------------------------------------
    if HDF5Dataset is not None:
        source_h5 = "/home/hasselnot/ML/SAM-Med3D/datasets/sample_A_padded_20160501.hdf"
        target_h5 = "/home/hasselnot/ML/SAM-Med3D/datasets/sample_A+_padded_20160601.hdf"
        
        # 定义数据集
        # 确保 crop_size 与 sam_vit_cfg['img_size'] 一致
        source_ds = HDF5Dataset(source_h5, crop_size=(current_img_size, current_img_size, current_img_size), mode='train')
        target_ds = HDF5Dataset(target_h5, crop_size=(current_img_size, current_img_size, current_img_size), mode='train')
        
        # 定义DataLoader
        source_loader = DataLoader(source_ds, batch_size=2, shuffle=True, num_workers=2)
        target_loader = DataLoader(target_ds, batch_size=2, shuffle=True, num_workers=2)
        val_loader = DataLoader(source_ds, batch_size=2, shuffle=False, num_workers=2) # 验证集
        
        print("Starting training with real data...")
        train_domain_adaptation(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
            val_loader=val_loader,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_epochs=100,
            lr=1e-4,
            save_dir='checkpoints'
        )
        exit()


