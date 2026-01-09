from medpy.metric.binary import hd95
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)


class DiceAverage(object):
    """二分类场景下的Dice计算（区分前景与背景）"""

    def __init__(self):
        self.class_num = 2  # 固定为二分类
        self.reset()

    def reset(self):
        # 分别存储背景（0）和前景（1）的指标
        self.tp = np.zeros(2, dtype=np.float64)  # [背景tp, 前景tp]
        self.fp = np.zeros(2, dtype=np.float64)
        self.fn = np.zeros(2, dtype=np.float64)
        self.epsilon = 1e-6

    def update(self, logits, targets):
        """
        计算二分类的Dice（区分前景和背景）
        logits: 模型输出 (B, 1, D, H, W) 或 (B, 2, D, H, W)
        targets: 标签 (B, 1, D, H, W) 单通道（0=背景，1=前景）
        """
        # 模型输出处理：单通道用sigmoid，双通道直接取概率
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)  # (B, 1, D, H, W)
            pred_mask = (probs > 0.5).float()  # 前景掩码（1=前景，0=背景）
        else:  # 双通道（背景+前景）
            probs = torch.softmax(logits, dim=1)  # (B, 2, D, H, W)
            pred_mask = torch.argmax(probs, dim=1, keepdim=True).float()  # (B, 1, D, H, W) 0=背景，1=前景

        # 确保标签和预测的维度一致
        if targets.dim() == 4:
            targets = targets.unsqueeze(1)  # (B, 1, D, H, W)
        if pred_mask.dim() == 4:
            pred_mask = pred_mask.unsqueeze(1)  # (B, 1, D, H, W)

        batch_size = logits.shape[0]

        for b in range(batch_size):
            current_target = targets[b, 0]  # 当前样本标签（0/1）
            current_pred = pred_mask[b, 0]  # 当前样本预测（0/1）

            # 分别计算背景（0）和前景（1）的tp/fp/fn
            for cls in [0, 1]:
                # 生成当前类别的掩码（1=当前类，0=其他）
                target_mask = (current_target == cls).float()
                pred_cls_mask = (current_pred == cls).float()

                # 转为布尔型计算交并
                target_mask = target_mask.bool()
                pred_cls_mask = pred_cls_mask.bool()

                # 累计指标
                self.tp[cls] += torch.sum(pred_cls_mask & target_mask).item()
                self.fp[cls] += torch.sum(pred_cls_mask & ~target_mask).item()
                self.fn[cls] += torch.sum(~pred_cls_mask & target_mask).item()

    def get_dice(self, cls):
        """计算指定类别的Dice（0=背景，1=前景）"""
        numerator = 2 * self.tp[cls] + self.epsilon
        denominator = 2 * self.tp[cls] + self.fp[cls] + self.fn[cls] + self.epsilon
        return numerator / denominator

    @property
    def avg(self):
        """返回背景和前景的平均Dice"""
        return [self.get_dice(0), self.get_dice(1)]


class SegmentationMetrics:
    def __init__(self, epsilon=1e-6):
        self.reset()
        self.epsilon = epsilon  # 用于数值稳定性

    def reset(self):
        self.tp = [0, 0]  # 背景和前景
        self.fp = [0, 0]
        self.fn = [0, 0]
        self.hd95_sum = [0, 0]
        self.hd95_count = [0, 0]
        self.sample_count = 0

    def update(self, pred, target, voxel_spacing=None):
        # 确保输入是有效的
        if pred is None or target is None:
            return

        # 转换为二值掩码
        if pred.shape[1] == 1:  # sigmoid
            pred_mask = (torch.sigmoid(pred) > 0.5).float()
        else:  # softmax
            pred_mask = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)

        # 确保维度一致
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # 按样本计算指标
        self.sample_count += pred.shape[0]
        for b in range(pred.shape[0]):
            # 按类别计算TP, FP, FN
            for cls in [0, 1]:  # 0:背景, 1:前景
                # 使用原地操作减少内存分配
                current_pred = (pred_mask[b, 0] == cls).bool().cpu().numpy()
                current_target = (target[b, 0] == cls).bool().cpu().numpy()

                # 计算交集和并集
                intersection = np.logical_and(current_pred, current_target)
                self.tp[cls] += intersection.sum()
                self.fp[cls] += np.logical_and(current_pred, ~current_target).sum()
                self.fn[cls] += np.logical_and(~current_pred, current_target).sum()

                # 仅当两个掩码都非空时计算HD95
                if current_target.any() and current_pred.any():
                    try:
                        spacing = voxel_spacing if voxel_spacing is not None else [1.0, 1.0, 1.0]
                        # 直接计算HD95而不存储中间变量
                        hd = hd95(
                            current_pred.astype(np.uint8),
                            current_target.astype(np.uint8),
                            voxelspacing=spacing
                        )
                        self.hd95_sum[cls] += hd
                        self.hd95_count[cls] += 1
                    except Exception as e:
                        print(f"Error calculating HD95 for class {cls}, sample {b}: {e}")

                # 及时释放内存
                del current_pred, current_target, intersection
                if 'hd' in locals():
                    del hd

    def get_dice(self, cls=1):
        """计算指定类别的Dice系数"""
        return (2 * self.tp[cls] + self.epsilon) / (2 * self.tp[cls] + self.fp[cls] + self.fn[cls] + self.epsilon)

    def get_iou(self, cls=1):
        """计算指定类别的IoU"""
        return (self.tp[cls] + self.epsilon) / (self.tp[cls] + self.fp[cls] + self.fn[cls] + self.epsilon)

    def get_hd95(self, cls=1):
        """计算指定类别的HD95"""
        if self.hd95_count[cls] > 0:
            return self.hd95_sum[cls] / self.hd95_count[cls]
        return np.nan

    @property
    def avg_dice(self):
        """返回背景和前景的平均Dice"""
        return (self.get_dice(0) + self.get_dice(1)) / 2

    def get_metrics(self):
        """返回所有指标的字典（含背景和前景）"""
        metrics = OrderedDict()
        # 计算背景和前景的指标
        for cls in [0, 1]:
            metrics[f'DSC_{cls}'] = round(self.get_dice(cls), 4)
            metrics[f'IoU_{cls}'] = round(self.get_iou(cls), 4)
            hd95 = self.get_hd95(cls)
            metrics[f'HD95_{cls}'] = round(hd95, 4) if not np.isnan(hd95) else np.nan

        # 计算均值
        metrics['Mean_DSC'] = round(np.mean([metrics[f'DSC_{cls}'] for cls in [0, 1]]), 4)
        metrics['Mean_IoU'] = round(np.mean([metrics[f'IoU_{cls}'] for cls in [0, 1]]), 4)

        hd_values = [metrics[f'HD95_{cls}'] for cls in [0, 1] if not np.isnan(metrics[f'HD95_{cls}'])]
        metrics['Mean_HD95'] = round(np.mean(hd_values), 4) if hd_values else np.nan

        # 样本统计
        metrics['Total_Samples'] = self.sample_count
        return metrics