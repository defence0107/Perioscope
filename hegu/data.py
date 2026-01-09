import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom, rotate
import random


def resize_3d(image, target_shape=(64, 64, 64)):
    """3D图像缩放，优化内存使用"""
    image = image.astype(np.float32)

    depth, height, width = image.shape
    scale_factors = (
        target_shape[0] / depth,
        target_shape[1] / height,
        target_shape[2] / width
    )

    # 分阶段缩放以减少内存峰值
    if depth > target_shape[0] * 2:
        mid_depth = max(target_shape[0] * 2, (depth + target_shape[0]) // 2)
        image = zoom(image, (mid_depth / depth, 1, 1), order=1)
        depth = mid_depth

    if height > target_shape[1] * 2:
        mid_height = max(target_shape[1] * 2, (height + target_shape[1]) // 2)
        image = zoom(image, (1, mid_height / height, 1), order=1)
        height = mid_height

    if width > target_shape[2] * 2:
        mid_width = max(target_shape[2] * 2, (width + target_shape[2]) // 2)
        image = zoom(image, (1, 1, mid_width / width), order=1)
        width = mid_width

    # 最终缩放到目标尺寸
    result = zoom(image, scale_factors, order=1)
    # 确保数组是连续的
    if not result.flags.contiguous:
        result = result.copy()
    return result


class MedicalImageAugmenter:
    """医学影像专用3D数据增强器"""

    def __init__(self,
                 rotation_range=15,
                 flip_prob=0.5,
                 noise_factor=0.01,
                 contrast_range=(0.8, 1.2),
                 shift_range=5):
        self.rotation_range = rotation_range  # 旋转角度范围
        self.flip_prob = flip_prob  # 翻转概率
        self.noise_factor = noise_factor  # 噪声强度因子
        self.contrast_range = contrast_range  # 对比度调整范围
        self.shift_range = shift_range  # 平移像素范围

    def __call__(self, image):
        """对3D图像应用随机增强（输入为numpy数组，形状为[C, D, H, W]）"""
        # 确保输入是numpy数组
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # 移除通道维度进行处理 [C, D, H, W] -> [D, H, W]
        data = image[0].copy()  # 显式复制以避免负步长问题

        # 1. 随机旋转（仅在两个平面上旋转，避免过度旋转影响解剖结构）
        if random.random() < 0.5:
            # 在冠状面旋转 (围绕D轴)
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            data = rotate(data, angle, axes=(1, 2), reshape=False, order=1, mode='constant')
        else:
            # 在矢状面旋转 (围绕H轴)
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            data = rotate(data, angle, axes=(0, 2), reshape=False, order=1, mode='constant')

        # 2. 随机翻转（三个平面中的一个）
        if random.random() < self.flip_prob:
            axis = random.choice([0, 1, 2])  # 随机选择一个轴进行翻转
            data = np.flip(data, axis=axis)
            # 翻转后可能产生负步长，确保数组连续
            if not data.flags.contiguous:
                data = data.copy()

        # 3. 随机平移（有限范围内，保持图像尺寸）
        if random.random() < 0.3:  # 30%概率应用平移
            shift_d = random.randint(-self.shift_range, self.shift_range)
            shift_h = random.randint(-self.shift_range, self.shift_range)
            shift_w = random.randint(-self.shift_range, self.shift_range)

            # 应用平移
            data = np.roll(data, shift_d, axis=0)
            data = np.roll(data, shift_h, axis=1)
            data = np.roll(data, shift_w, axis=2)

            # 填充平移产生的空白区域（使用边缘值）
            if shift_d > 0:
                data[:shift_d, :, :] = data[shift_d, :, :].mean()
            elif shift_d < 0:
                data[shift_d:, :, :] = data[shift_d - 1, :, :].mean()

            if shift_h > 0:
                data[:, :shift_h, :] = data[:, shift_h, :].mean()
            elif shift_h < 0:
                data[:, shift_h:, :] = data[:, shift_h - 1, :].mean()

            if shift_w > 0:
                data[:, :, :shift_w] = data[:, :, shift_w].mean()
            elif shift_w < 0:
                data[:, :, shift_w:] = data[:, :, shift_w - 1].mean()

        # 4. 随机噪声（高斯噪声）
        if random.random() < 0.3:  # 30%概率添加噪声
            noise = np.random.normal(0, self.noise_factor * data.std(), data.shape)
            data = data + noise
            # 确保值在合理范围内
            data = np.clip(data, data.min(), data.max())

        # 5. 随机对比度调整
        if random.random() < 0.3:  # 30%概率调整对比度
            factor = random.uniform(*self.contrast_range)
            mean = data.mean()
            data = (data - mean) * factor + mean
            # 确保值在合理范围内
            data = np.clip(data, data.min(), data.max())

        # 确保数组是连续的
        if not data.flags.contiguous:
            data = data.copy()

        # 恢复通道维度 [D, H, W] -> [C, D, H, W]
        augmented = np.expand_dims(data, axis=0)
        return torch.from_numpy(augmented).float()


class MedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_shape=(64, 64, 64), is_train=True):
        """处理3D NIfTI数据的Dataset，支持数据增强"""
        self.classes = ['less_than_1_3', '1_3_to_2_3', 'more_than_2_3']
        self.file_list = []
        self.labels = []
        self.target_shape = target_shape
        self.is_train = is_train  # 标记是否为训练集（训练集才应用增强）

        # 基础变换：标准化
        self.base_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 训练集增强变换
        self.augmenter = MedicalImageAugmenter(
            rotation_range=15,
            flip_prob=0.5,
            noise_factor=0.01,
            contrast_range=(0.8, 1.2),
            shift_range=5
        )

        # 如果指定了外部transform，与内部变换组合
        if transform:
            self.external_transform = transform
        else:
            self.external_transform = None

        # 扫描所有文件
        self._scan_files(root_dir)

    def _scan_files(self, root_dir):
        """扫描所有NIfTI文件"""
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 类别目录 {class_dir} 不存在")
                continue

            for root, _, files in os.walk(class_dir):
                for filename in files:
                    if filename.lower().endswith('.nii') or filename.lower().endswith('.nii.gz'):
                        file_path = os.path.join(root, filename)
                        self.file_list.append(file_path)
                        self.labels.append(class_idx)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        try:
            data = self._load_nifti(file_path)

            # 应用数据增强（仅训练集）
            if self.is_train:
                data = self.augmenter(data)

            # 应用基础变换（标准化）
            data = self.base_transform(data)

            # 应用外部变换（如果有）
            if self.external_transform:
                data = self.external_transform(data)

            return data, label
        except MemoryError:
            print(f"处理文件 {file_path} 时内存不足，尝试减小目标尺寸")
            # 尝试使用更小的尺寸重新处理
            small_data = self._load_nifti(file_path, force_smaller=True)

            # 对小尺寸数据应用增强和变换
            if self.is_train:
                small_data = self.augmenter(small_data)
            small_data = self.base_transform(small_data)
            if self.external_transform:
                small_data = self.external_transform(small_data)

            return small_data, label
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            # 返回一个空数据和对应标签，允许程序继续运行
            empty_data = torch.zeros((1,) + self.target_shape, dtype=torch.float32)
            return empty_data, label

    def _load_nifti(self, file_path, force_smaller=False):
        """加载并预处理NIfTI文件，优化内存使用"""
        # 使用内存映射方式加载大型文件
        img = nib.load(file_path, mmap=True)

        # 根据需要强制使用更小的目标尺寸
        target_shape = (32, 32, 32) if force_smaller else self.target_shape

        # 逐步加载和处理数据以减少内存占用
        data = img.get_fdata(dtype=np.float32)  # 使用float32而不是默认的float64

        # 确保数组是连续的
        if not data.flags.contiguous:
            data = data.copy()

        # 确保3D
        if len(data.shape) != 3:
            # 尝试降维到3D
            if len(data.shape) == 4:
                if data.shape[3] == 1:
                    data = data[:, :, :, 0]
                else:
                    # 取第一个通道
                    data = data[:, :, :, 0]
                    print(f"警告: {file_path} 是4D数据，已取第一个通道")
            else:
                raise ValueError(f"需要3D数据，但获取到 {len(data.shape)}D 数据")

        # 调整维度顺序 (D, H, W)
        data = np.transpose(data, (2, 0, 1))

        # 检查并处理负步长
        if not data.flags.contiguous:
            data = data.copy()

        # 3D resize
        data = resize_3d(data, target_shape)

        # 添加通道维度 (1, D, H, W)
        data = np.expand_dims(data, axis=0)

        # 清理内存
        del img

        return torch.from_numpy(data).float()


# 使用示例
if __name__ == '__main__':
    # 训练集（启用增强）
    train_dataset = MedicalDataset(
        root_dir='D:/project/raw_dataset/hegus',
        target_shape=(64, 64, 64),
        is_train=True
    )

    # 验证集（不启用增强）
    val_dataset = MedicalDataset(
        root_dir='D:/project/raw_dataset/hegus',
        target_shape=(64, 64, 64),
        is_train=False
    )

    # 测试数据加载
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)