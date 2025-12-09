import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    """
    针对 HDF5 格式体电镜数据的 Dataset
    数据结构:
    - volumes/raw: 原始灰度图像 (D, H, W)
    - volumes/labels/clefts: 突触间隙标签 (D, H, W)
    """
    def __init__(self, h5_path, crop_size=(16, 128, 128), mode='train', transform=None):
        """
        Args:
            h5_path (str): HDF5 文件路径
            crop_size (tuple): 随机裁剪的 3D patch 大小 (D, H, W)
            mode (str): 'train' 或 'val' (目前主要影响是否随机裁剪)
            transform (callable, optional): 数据增强
        """
        self.h5_path = h5_path
        self.crop_size = crop_size
        self.mode = mode
        self.transform = transform
        
        # 打开 HDF5 文件读取元数据
        # 注意：在 __init__ 中打开文件可能会导致多进程 DataLoader 报错，
        # 建议在 __getitem__ 中打开，或者使用 worker_init_fn
        # 这里为了简单，先读取 shape，具体数据在 getitem 读取
        with h5py.File(self.h5_path, 'r') as f:
            self.raw_shape = f['volumes/raw'].shape
            # 检查是否有标签
            self.has_label = 'volumes/labels/clefts' in f
            
        print(f"Loaded HDF5: {h5_path}, Shape: {self.raw_shape}, Has Label: {self.has_label}")

    def __len__(self):
        # 对于大体积数据，epoch 的长度可以定义为采样的 patch 数量
        # 这里简单定义为 1000 个 patch 一个 epoch，可根据需要调整
        return 1000

    def __getitem__(self, index):
        # 在 getitem 中打开文件，确保多进程安全
        with h5py.File(self.h5_path, 'r') as f:
            # 1. 随机采样坐标
            # 确保裁剪框在图像范围内
            d, h, w = self.raw_shape
            cd, ch, cw = self.crop_size
            
            # 随机选择起始点
            z = np.random.randint(0, d - cd)
            y = np.random.randint(0, h - ch)
            x = np.random.randint(0, w - cw)
            
            # 2. 读取数据 patch
            # HDF5 支持切片读取，只读取需要的 patch，不用加载整个卷
            raw_patch = f['volumes/raw'][z:z+cd, y:y+ch, x:x+cw]
            
            if self.has_label:
                label_patch = f['volumes/labels/clefts'][z:z+cd, y:y+ch, x:x+cw]
            else:
                # 如果没有标签（如目标域数据），生成全0占位
                label_patch = np.zeros(self.crop_size, dtype=np.uint8)

        # 3. 数据预处理
        # 归一化到 [0, 1] 并转为 float32
        raw_patch = raw_patch.astype(np.float32) / 255.0
        
        # 增加通道维度 (D, H, W) -> (C=1, D, H, W)
        raw_patch = np.expand_dims(raw_patch, axis=0)
        
        # 处理标签
        label_patch = label_patch.astype(np.float32)
        # 二值化：假设标签非0即为前景
        label_patch = (label_patch > 0).astype(np.float32)
        label_patch = np.expand_dims(label_patch, axis=0)

        # 转为 Tensor
        img_tensor = torch.from_numpy(raw_patch)
        label_tensor = torch.from_numpy(label_patch)

        # 应用数据增强 (如果有)
        if self.transform:
            # 注意：需确保 transform 能同时处理 3D img 和 label
            pass

        return img_tensor, label_tensor

# -----------------------------------------------------------------------------
# 配对数据集 (用于无监督域适应 Phase 2)
# -----------------------------------------------------------------------------
class PairedHDF5Dataset(Dataset):
    """
    同时从源域和目标域采样，用于 Phase 2 训练
    假设源域和目标域是未配对的 (Unpaired)，随机采样
    """
    def __init__(self, source_h5, target_h5, crop_size=(16, 128, 128)):
        self.source_ds = HDF5Dataset(source_h5, crop_size=crop_size)
        self.target_ds = HDF5Dataset(target_h5, crop_size=crop_size)
        
    def __len__(self):
        return len(self.source_ds)
        
    def __getitem__(self, index):
        s_img, s_label = self.source_ds[index]
        t_img, _ = self.target_ds[index] # 目标域忽略标签
        return s_img, s_label, t_img

if __name__ == "__main__":
    # 简易测试
    # 需要一个真实的 h5 文件路径才能运行
    print("HDF5Dataset defined. Please provide .h5 file path to test.")
