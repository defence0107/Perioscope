import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom


def resize_3d(image, target_shape=(128, 128, 128)):
    """3D图像缩放"""
    depth, height, width = image.shape
    scale_factors = (
        target_shape[0] / depth,
        target_shape[1] / height,
        target_shape[2] / width
    )
    return zoom(image, scale_factors)


class MedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """仅处理3D NIfTI数据的Dataset"""
        self.classes = ['less_than_1_3', '1_3_to_2_3', 'more_than_2_3']
        self.file_list = []
        self.labels = []
        self.transform = transform


        # 扫描NIfTI文件
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for root, _, files in os.walk(class_dir):
                for filename in files:
                    if filename.lower().endswith('.nii'):
                        file_path = os.path.join(root, filename)
                        self.file_list.append(file_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        data = self._load_nifti(file_path)

        if self.transform:
            data = self.transform(data)

        return data, int(label)

    def _load_nifti(self, file_path):
        """加载并预处理NIfTI文件"""
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)

        # 确保3D
        if len(data.shape) != 3:
            raise ValueError(f"需要3D数据，但获取到 {len(data.shape)}D 数据")

        # 调整维度顺序 (D, H, W)
        data = np.transpose(data, (2, 0, 1))

        # 3D resize
        data = resize_3d(data)

        # 添加通道维度 (1, D, H, W)
        data = np.expand_dims(data, axis=0)

        return torch.from_numpy(data).float()


# Transform现在只需要处理标准化
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 使用示例
if __name__ == '__main__':
    dataset = MedicalDataset(root_dir='E:/D/data_set/hegu', transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for data, targets in dataloader:
        print(f'数据维度: {data.shape}')  # 应为 [batch, 1, 128, 128, 128]
        print(f'标签: {targets}')
        break