import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=100.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(
            input, target.float(),
            pos_weight=self.pos_weight
        )

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        二分类Dice Loss
        参数:
            pred: 模型输出 (N,1,D,H,W) 或 (N,D,H,W) 未经sigmoid
            target: 目标标签 (N,1,D,H,W) 或 (N,D,H,W) 值在[0,1]
        """
        # 统一维度
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)  # (N,D,H,W) -> (N,1,D,H,W)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        pred = torch.sigmoid(pred)  # 确保概率值
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        二分类Dice+BCE组合损失
        参数:
            pred: 模型原始输出 (N,1,D,H,W) 或 (N,D,H,W)
            target: 目标标签 (N,1,D,H,W) 或 (N,D,H,W) 值在[0,1]
        """
        # BCE部分 (自动处理sigmoid)
        bce_loss = self.bce(pred, target.float())

        # Dice部分
        pred_sig = torch.sigmoid(pred)
        if pred_sig.dim() == 4:
            pred_sig = pred_sig.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        intersection = (pred_sig * target).sum()
        union = pred_sig.sum() + target.sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


import torch
import torch.nn as nn

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        # 如果目标是单通道二值掩码，转换为类别索引
        if targets.size(1) == 1:
            targets = targets.squeeze(1).long()  # [B,D,H,W]

        # 计算softmax得到概率
        probs = F.softmax(logits, dim=1)

        # 提取每个类别的概率
        bg_prob = probs[:, 0]
        fg_prob = probs[:, 1]

        # 创建每个类别的目标掩码
        bg_target = (targets == 0).float()
        fg_target = (targets == 1).float()

        # 计算真正类、假正类、假负类
        tp = (fg_prob * fg_target).sum()
        fp = (fg_prob * bg_target).sum()
        fn = ((1 - fg_prob) * fg_target).sum()

        # 计算Tversky指数 n
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + self.eps)

        # 返回损失
        return 1 - tversky


class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        tversky_loss = self.tversky(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * tversky_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, dice_weight=0.5):
        """
        二分类Focal Loss + Dice Loss组合
        """
        super(FocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # Focal Loss部分
        pred_sig = torch.sigmoid(pred)
        if pred_sig.dim() == 4:
            pred_sig = pred_sig.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce).mean()

        # Dice Loss部分
        dice_loss = self.dice(pred, target)

        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss

# 使用示例
if __name__ == "__main__":
    # 模拟数据
    pred = torch.randn(2, 1, 32, 32, 32)  # 模型原始输出
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()  # 二分类标签

    # 初始化损失函数
    dice_loss = DiceLoss()
    dice_bce_loss = DiceBCELoss(dice_weight=0.7)
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
    focal_dice_loss = FocalDiceLoss()

    # 计算损失
    print("Dice Loss:", dice_loss(pred, target).item())
    print("Dice+BCE Loss:", dice_bce_loss(pred, target).item())
    print("Tversky Loss:", tversky_loss(pred, target).item())
    print("Focal+Dice Loss:", focal_dice_loss(pred, target).item())