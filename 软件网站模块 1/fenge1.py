from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk
from torch.nn.parallel import DataParallel


class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args, device='cuda'):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride
        self.device = torch.device(device)

        # 确保cut_size和cut_stride是三元组
        if isinstance(self.cut_size, int):
            self.cut_size = (self.cut_size, self.cut_size, self.cut_size)
        if isinstance(self.cut_stride, int):
            self.cut_stride = (self.cut_stride, self.cut_stride, self.cut_stride)

        # 读取和处理CT数据
        self.ct = sitk.ReadImage(data_path, sitk.sitkInt16)
        self.data_np = sitk.GetArrayFromImage(self.ct)
        self.ori_shape = self.data_np.shape
        print(f"Original image shape: {self.ori_shape}")

        # 调整大小和归一化（CPU操作）
        self.data_np = ndimage.zoom(self.data_np,
                                    (512 / self.ori_shape[0],
                                     256 / self.ori_shape[1],
                                     256 / self.ori_shape[2]),
                                    order=3)
        self.data_np = np.clip(self.data_np, args.lower, args.upper)
        self.data_np = self.data_np / args.norm_factor
        self.resized_shape = self.data_np.shape
        print(f"Resized image shape: {self.resized_shape}")

        # 填充和补丁提取（仍在CPU上）
        self.data_np = self.padding_img(self.data_np, self.cut_size, self.cut_stride)
        self.padding_shape = self.data_np.shape
        print(f"Padded image shape: {self.padding_shape}")

        self.patches = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)
        print(f"Patches number of the image: {len(self.patches)}")

        # 处理标签并立即移至GPU
        self.seg = sitk.ReadImage(label_path, sitk.sitkInt8)
        self.label_np = sitk.GetArrayFromImage(self.seg)
        if self.n_labels == 2:
            self.label_np[self.label_np > 0] = 1
        self.label = torch.from_numpy(np.expand_dims(self.label_np, axis=0)).long().to(self.device)

        # 在GPU上初始化结果存储
        self.result = None

    def __getitem__(self, index):
        # 加载后立即将数据移至GPU
        # 增加详细的索引检查和错误信息
        if index < 0 or index >= len(self.patches):
            raise IndexError(
                f"索引 {index} 超出范围！有效范围是 [0, {len(self.patches) - 1}], 共 {len(self.patches)} 个补丁"
            )

        # 获取补丁并转换为张量
        patch = self.patches[index]
        data = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)
        return data

    def __len__(self):
        # 返回补丁的实际数量
        return len(self.patches) if hasattr(self, 'patches') else 0

    def update_result(self, tensor):
        # 确保张量保持在GPU上
        if not tensor.is_cuda:
            tensor = tensor.to(self.device)
            print("警告：输入张量不在GPU上，已自动转移")

        # 获取输入张量形状并计算填充
        # 调整维度索引以匹配实际输出形状
        if len(tensor.shape) != 5:  # 期望形状: [batch, channels, depth, height, width]
            raise ValueError(f"张量形状不正确，期望5维，实际{len(tensor.shape)}维: {tensor.shape}")

        batch_size, channels, patch_d, patch_h, patch_w = tensor.shape

        # 计算需要的填充
        padding_needed_d = self.cut_size[0] - patch_d
        padding_needed_h = self.cut_size[1] - patch_h
        padding_needed_w = self.cut_size[2] - patch_w

        # 在GPU上应用填充
        padded_tensor = torch.nn.functional.pad(
            tensor,
            (0, padding_needed_w, 0, padding_needed_h, 0, padding_needed_d)  # 注意填充顺序与维度对应
        ).to(self.device)

        # 在GPU上存储结果
        if self.result is None:
            self.result = padded_tensor
        else:
            # 确保拼接前设备一致
            if self.result.device != padded_tensor.device:
                padded_tensor = padded_tensor.to(self.result.device)
            self.result = torch.cat((self.result, padded_tensor), dim=0)

    def recompone_result(self):
        if self.result is None:
            raise ValueError("结果为空，请先运行update_result填充结果")

        # 计算补丁数量和重组参数
        patch_d, patch_h, patch_w = self.cut_size
        n_patches_d = (self.padding_shape[0] - patch_d) // self.cut_stride[0] + 1
        n_patches_h = (self.padding_shape[1] - patch_h) // self.cut_stride[1] + 1
        n_patches_w = (self.padding_shape[2] - patch_w) // self.cut_stride[2] + 1

        total_patches = n_patches_d * n_patches_h * n_patches_w

        # 验证补丁数量是否匹配
        if self.result.shape[0] != total_patches:
            raise ValueError(
                f"补丁数量不匹配！期望 {total_patches} 个，实际 {self.result.shape[0]} 个"
            )

        # 初始化GPU上的张量
        full_prob = torch.zeros(
            (self.n_labels, self.padding_shape[0], self.padding_shape[1], self.padding_shape[2]),
            device=self.device
        )
        full_sum = torch.zeros_like(full_prob, device=self.device)

        # 在GPU上累积结果
        patch_idx = 0
        for d in range(n_patches_d):
            for h in range(n_patches_h):
                for w in range(n_patches_w):
                    # 计算当前补丁的位置
                    start_d = d * self.cut_stride[0]
                    end_d = start_d + patch_d
                    start_h = h * self.cut_stride[1]
                    end_h = start_h + patch_h
                    start_w = w * self.cut_stride[2]
                    end_w = start_w + patch_w

                    # 确保索引在有效范围内
                    if end_d > self.padding_shape[0]:
                        end_d = self.padding_shape[0]
                        start_d = end_d - patch_d
                    if end_h > self.padding_shape[1]:
                        end_h = self.padding_shape[1]
                        start_h = end_h - patch_h
                    if end_w > self.padding_shape[2]:
                        end_w = self.padding_shape[2]
                        start_w = end_w - patch_w

                    # 累积结果
                    full_prob[:, start_d:end_d, start_h:end_h, start_w:end_w] += self.result[patch_idx]
                    full_sum[:, start_d:end_d, start_h:end_h, start_w:end_w] += 1
                    patch_idx += 1

        # 处理可能的零值（避免除零）
        full_sum = torch.clamp(full_sum, min=1.0)

        # 最终平均并限制范围
        final_avg = full_prob / full_sum
        final_avg = torch.clamp(final_avg, min=0.0, max=1.0)

        # 裁剪回原始形状
        img = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img.unsqueeze(0)

    def padding_img(self, img, size, stride):
        assert len(img.shape) == 3  # 3D array

        img_d, img_h, img_w = img.shape

        # 计算每个维度需要的填充量
        leftover_d = (img_d - size[0]) % stride[0]
        leftover_h = (img_h - size[1]) % stride[1]
        leftover_w = (img_w - size[2]) % stride[2]

        padding_d = stride[0] - leftover_d if leftover_d != 0 else 0
        padding_h = stride[1] - leftover_h if leftover_h != 0 else 0
        padding_w = stride[2] - leftover_w if leftover_w != 0 else 0

        # 填充图像
        padded_img = np.pad(
            img,
            ((0, padding_d), (0, padding_h), (0, padding_w)),
            mode='constant',
            constant_values=0
        )

        print(f"Padded shape: {padded_img.shape}, original shape: {img.shape}")
        return padded_img

    def extract_ordered_overlap(self, img, size, stride):
        img_d, img_h, img_w = img.shape

        # 计算每个维度的patch数量
        n_patches_d = (img_d - size[0]) // stride[0] + 1
        n_patches_h = (img_h - size[1]) // stride[1] + 1
        n_patches_w = (img_w - size[2]) // stride[2] + 1

        total_patches = n_patches_d * n_patches_h * n_patches_w
        print(f"Extracting {total_patches} patches: {n_patches_d}x{n_patches_h}x{n_patches_w}")

        # 预分配补丁数组
        patches = np.empty((total_patches, size[0], size[1], size[2]), dtype=np.float32)

        patch_idx = 0
        for d in range(n_patches_d):
            for h in range(n_patches_h):
                for w in range(n_patches_w):
                    # 计算当前patch的位置
                    start_d = d * stride[0]
                    end_d = start_d + size[0]
                    start_h = h * stride[1]
                    end_h = start_h + size[1]
                    start_w = w * stride[2]
                    end_w = start_w + size[2]

                    # 确保不会超出边界
                    if end_d > img_d:
                        end_d = img_d
                        start_d = end_d - size[0]
                    if end_h > img_h:
                        end_h = img_h
                        start_h = end_h - size[1]
                    if end_w > img_w:
                        end_w = img_w
                        start_w = end_w - size[2]

                    # 提取patch
                    patch = img[start_d:end_d, start_h:end_h, start_w:end_w]
                    patches[patch_idx] = patch
                    patch_idx += 1

        return patches


def Eval_Datasets (dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'ct/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test samples is: ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, labelpath, args=args, device='cuda'), datapath.split('-')[-1]