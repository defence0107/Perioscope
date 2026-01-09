"""
This part is based on the dataset class implemented by pytorch,
including train_dataset and test_dataset, as well as data augmentation
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

#----------------------data augment-------------------------------------------
class Resize:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, mask):
        # 直接在原始张量上操作，避免创建新的张量
        img = F.interpolate(img.unsqueeze(0), scale_factor=(1, self.scale, self.scale),
                            mode='trilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), scale_factor=(1, self.scale, self.scale),
                             mode="trilinear", recompute_scale_factor=True).squeeze(0).long()
        return img, mask


class RandomResize:
    def __init__(self, s_rank, w_rank, h_rank):
        # 确保传入的参数是元组或列表类型
        if not isinstance(w_rank, (tuple, list)):
            w_rank = (w_rank, w_rank)
        if not isinstance(h_rank, (tuple, list)):
            h_rank = (h_rank, h_rank)
        if not isinstance(s_rank, (tuple, list)):
            s_rank = (s_rank, s_rank)

        self.w_rank = w_rank
        self.h_rank = h_rank
        self.s_rank = s_rank

    def __call__(self, img, mask):
        # 从指定范围中随机选择数值
        random_w = random.randint(self.w_rank[0], self.w_rank[1])
        random_h = random.randint(self.h_rank[0], self.h_rank[1])
        random_s = random.randint(self.s_rank[0], self.s_rank[1])
        shape = [random_s, random_h, random_w]

        # 对图像和掩码进行插值操作
        img = F.interpolate(img.unsqueeze(0), size=shape,
                            mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=shape,
                             mode="trilinear").squeeze(0).long()
        return img, mask

class RandomCrop:
    def __init__(self, slices, with_label_prob=0.8):
        """
        以标签区域为中心进行随机裁剪

        参数:
            slices: 期望的裁剪深度
            with_label_prob: 裁剪区域包含标签的概率 (0.0-1.0)
        """
        self.slices = slices
        self.with_label_prob = with_label_prob

    def _get_range(self, total_slices, crop_slices, label_start=None, label_end=None):
        """
        计算随机裁剪的范围，优先包含标签区域

        参数:
            total_slices: 原始图像总深度
            crop_slices: 目标裁剪深度
            label_start: 标签开始位置
            label_end: 标签结束位置
        """
        # 如果没有标签或随机决定不关注标签，进行完全随机裁剪
        if label_start is None or label_end is None or random.random() > self.with_label_prob:
            if total_slices < crop_slices:
                return 0, total_slices
            start = random.randint(0, total_slices - crop_slices)
            return start, start + crop_slices

        # 标签区域的长度
        label_length = label_end - label_start

        # 如果标签区域足够大，直接在标签区域内随机裁剪
        if label_length >= crop_slices:
            start = random.randint(label_start, label_end - crop_slices)
            return start, start + crop_slices

        # 标签区域较小，尝试以标签为中心进行裁剪
        center = (label_start + label_end) // 2
        start = max(0, center - crop_slices // 2)
        end = start + crop_slices

        # 如果裁剪范围超出图像边界，调整位置
        if end > total_slices:
            end = total_slices
            start = max(0, end - crop_slices)

        return start, end

    def __call__(self, img, mask):
        """执行裁剪，确保优先包含标签区域"""
        total_slices = mask.size(1)

        # 计算标签在深度维度上的范围
        label_slices = (mask.sum(dim=(0, 2, 3)) > 0).nonzero(as_tuple=True)[0]
        label_start = label_slices.min().item() if label_slices.numel() > 0 else None
        label_end = label_slices.max().item() + 1 if label_slices.numel() > 0 else None

        # 获取裁剪范围
        ss, es = self._get_range(total_slices, self.slices, label_start, label_end)

        # 直接切片，避免创建零张量
        return img[:, ss:es], mask[:, ss:es]

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            # 使用原地操作
            img = torch.flip(img, [2])
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            # 使用原地操作
            img = torch.flip(img, [3])
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        # 使用原地操作
        img = torch.rot90(img, cnt, [2, 3])  # 修正维度索引
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0, self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)

class Center_Crop:
    def __init__(self, base=16, max_size=None):
        self.base = base  # 基础单位，默认为16，因为4次下采样后为1
        self.max_size = max_size
        # 确保max_size是base的倍数
        if self.max_size is not None and self.max_size % self.base != 0:
            self.max_size = self.max_size - (self.max_size % self.base)

    def __call__(self, img, label):
        # 检查图像维度是否符合要求
        if img.size(1) < self.base:
            return img, label  # 返回原始图像而不是None

        # 计算标签中非零区域的起始和结束位置
        non_zero_indices = (label.sum(dim=(0, 2, 3)) > 0).nonzero(as_tuple=True)[0]
        if len(non_zero_indices) == 0:
            # 如果没有非零区域，使用图像中心
            center = img.size(1) // 2
        else:
            # 计算非零区域的中心
            start_idx = non_zero_indices.min().item()
            end_idx = non_zero_indices.max().item()
            center = (start_idx + end_idx) // 2

        # 计算裁剪的切片数量
        slice_num = img.size(1) - (img.size(1) % self.base)
        if self.max_size is not None:
            slice_num = min(self.max_size, slice_num)

        # 计算裁剪的左右边界
        half_slice = slice_num // 2
        left = max(0, center - half_slice)
        right = left + slice_num

        # 调整边界以确保不超出图像范围
        if right > img.size(1):
            right = img.size(1)
            left = right - slice_num
        if left < 0:
            left = 0
            right = left + slice_num

        # 执行裁剪
        return img[:, left:right], label[:, left:right]

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))[None]  # 合并操作，减少中间变量
        return img, mask

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        # 使用原地归一化，设置inplace=True
        return normalize(img, self.mean, self.std, inplace=True), mask

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask