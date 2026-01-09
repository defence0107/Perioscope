from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, RandomResize
import matplotlib.pyplot as plt


class LabelBrightnessAdjust(object):
    """对标签进行"亮度增强"（例如调整特定类别的权重）"""

    def __init__(self, prob=0.5, scale_range=(0.8, 1.2)):
        self.prob = prob  # 应用此变换的概率
        self.scale_range = scale_range  # 亮度调整的范围

    def __call__(self, image, label):
        if random.random() > self.prob:
            return image, label

        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        # 只对前景区域进行调整（假设标签中的1表示前景）
        if isinstance(label, torch.Tensor):
            if label.dtype != torch.float32:
                label = label.float()  # 转换为浮点数以便进行乘法操作
            label[label > 0] *= scale  # 原地操作
        elif isinstance(label, np.ndarray):
            label = label.astype(np.float32, copy=False)
            label[label > 0] *= scale

        return image, label


# 优化Train_Dataset类以减少内存占用
class Train_Dataset(dataset):
    def __init__(self, args):
        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        # 添加LabelBrightnessAdjust到变换列表
        self.transforms = Compose([
            RandomCrop(self.args.crop_size),
            RandomFlip_LR(prob=0.5),
            RandomFlip_UD(prob=0.5),
            RandomResize(48, 256, 256),
            LabelBrightnessAdjust(prob=0.3)  # 30%的概率应用标签亮度调整
        ])

    def __getitem__(self, index):
        # 直接加载为numpy数组，减少中间转换步骤
        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        # 使用GetArrayFromImage确保数据可写，并立即指定为float32
        ct_array = sitk.GetArrayFromImage(ct).astype(np.float32, copy=False)
        seg_array = sitk.GetArrayFromImage(seg).astype(np.float32, copy=False)

        # 直接在float32数组上进行归一化，减少内存占用
        ct_array /= np.float32(self.args.norm_factor)  # 原地操作

        # 直接创建张量，避免中间numpy数组的内存占用
        ct_tensor = torch.from_numpy(ct_array).unsqueeze(0)
        seg_tensor = torch.from_numpy(seg_array).unsqueeze(0)

        if self.transforms:
            ct_tensor, seg_tensor = self.transforms(ct_tensor, seg_tensor)

        return ct_tensor, seg_tensor.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list


# 优化后的可视化代码，减少内存占用
if __name__ == "__main__":
    sys.path.append('D:/project/3DUNet2')
    from config import args

    # 设置DataLoader参数以优化内存使用
    train_ds = Train_Dataset(args)
    train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,  # 锁页内存，加速GPU数据传输
        persistent_workers=True  # 保持工作进程 alive，减少创建开销
    )

    for i, (ct, seg) in enumerate(train_dl):
        print(f"样本 {i}：")
        print(f"CT尺寸：{ct.shape}，标签尺寸：{seg.shape}")
        print(f"标签值范围：{seg.min().item()} ~ {seg.max().item()}")
        print(f"前景像素数：{torch.sum(seg).item()}，背景像素数：{seg.numel() - torch.sum(seg).item()}")

        # 获取CT的深度维度
        depth_dim = ct.shape[2]
        slice_idx = depth_dim // 2

        # 可视化时只复制需要的切片，而不是整个体积
        # 使用 detach() 减少计算图内存占用
        ct_slice = ct[0, 0, slice_idx].detach().cpu().numpy()
        seg_slice = seg[0, slice_idx].detach().cpu().numpy()

        # 可视化
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(ct_slice, cmap='gray')
        plt.title("CT图像")
        plt.subplot(122)
        plt.imshow(seg_slice, cmap='gray')
        plt.title("标签（0=背景，1=前景）")
        plt.tight_layout()  # 优化布局
        plt.show()

        # 显式释放不再使用的张量内存
        del ct, seg, ct_slice, seg_slice
        torch.cuda.empty_cache()  # 清理GPU缓存

        if i == 2:  # 只看3个样本
            break