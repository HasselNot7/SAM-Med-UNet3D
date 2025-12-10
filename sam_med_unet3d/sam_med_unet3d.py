import torch
import torch.nn as nn
from torch.nn import functional as F
from unet3d.unet3d import UNet3D, Conv3D_Block
from segment_anything.modeling.image_encoder3D import ImageEncoderViT3D


class Conv3DProjector(nn.Module):
    """三层3D卷积嵌入投影器"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class SAMMedUNet3D(nn.Module):
    """
    SAM-Med3D ViT编码器 + 三层卷积投影 + UNet3D特征融合网络
    """

    def __init__(self,
                 sam_vit_cfg: dict,
                 unet3d_cfg: dict,
                 projector_out_channels: int = 1024):
        super().__init__()
        self.sam_encoder = ImageEncoderViT3D(**sam_vit_cfg)
        self.unet3d = UNet3D(**unet3d_cfg)
        self.mix_block = Conv3D_Block(2048, 1024, residual='conv')
        self.projector_out_channels = projector_out_channels
        self.projector = None

    def forward(self, x):
        # x: (B, C, D, H, W)
        vit_feat = self.sam_encoder(x)  # (B, D', H', W', C)
        vit_feat = vit_feat.permute(0, 4, 1, 2, 3).contiguous()

        # 动态创建projector，确保输入通道数正确
        if self.projector is None or self.projector.proj[0].in_channels != vit_feat.shape[1]:
            self.projector = Conv3DProjector(
                vit_feat.shape[1], self.projector_out_channels).to(vit_feat.device)
        vit_proj = self.projector(vit_feat)

        x1 = self.unet3d.conv_blk1(x)
        x_low1 = self.unet3d.pool1(x1)
        x2 = self.unet3d.conv_blk2(x_low1)
        x_low2 = self.unet3d.pool2(x2)
        x3 = self.unet3d.conv_blk3(x_low2)
        x_low3 = self.unet3d.pool3(x3)
        x4 = self.unet3d.conv_blk4(x_low3)
        x_low4 = self.unet3d.pool4(x4)
        base = self.unet3d.conv_blk5(x_low4)

        if vit_proj.shape[2:] != base.shape[2:]:
            vit_proj = F.interpolate(
                vit_proj, size=base.shape[2:], mode='trilinear', align_corners=False)
        fused = torch.cat([base, vit_proj], dim=1)
        mixed = self.mix_block(fused)

        d4 = torch.cat([self.unet3d.deconv_blk4(mixed), x4], dim=1)
        d_high4 = self.unet3d.dec_conv_blk4(d4)
        d3 = torch.cat([self.unet3d.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.unet3d.dec_conv_blk3(d3)
        d2 = torch.cat([self.unet3d.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.unet3d.dec_conv_blk2(d2)
        d1 = torch.cat([self.unet3d.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.unet3d.dec_conv_blk1(d1)
        seg = self.unet3d.sigmoid(self.unet3d.one_conv(d_high1))
        return seg


if __name__ == "__main__":
    # 简易可用性测试
    sam_vit_cfg = dict(
        img_size=128,  # 假设输入为128^3
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    unet3d_cfg = dict(
        num_channels=1,
        feat_channels=[64, 128, 256, 512, 1024],
        residual=None
    )
    model = SAMMedUNet3D(sam_vit_cfg, unet3d_cfg, projector_out_channels=1024)
    model.eval()
    # 构造一个假输入 (B, C, D, H, W)
    x = torch.randn(2, 1, 128, 128, 128)
    with torch.no_grad():
        out = model(x)
    print('Input shape:', x.shape)
    print('Output shape:', out.shape)
