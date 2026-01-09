import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DepthwiseSeparableConv3d(nn.Module):
    """深度可分离卷积，增强初始化和稳定性"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        # 深度卷积（逐通道卷积）
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False  # 禁用偏置（批归一化会处理）
        )
        # 逐点卷积（1x1x1调整通道）
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        # 批归一化层（增强稳定性）
        self.bn = nn.BatchNorm3d(out_channels)

        # 权重初始化
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)  # 卷积后立即批归一化
        return x


class RSU3D(nn.Module):
    """残差U型单元，优化梯度流动"""

    def __init__(self, in_ch=3, mid_ch=8, out_ch=3, height=3):
        super().__init__()
        assert height >= 2, "height必须至少为2"

        # 输入卷积+激活（替换ReLU为LeakyReLU避免梯度死亡）
        self.conv_in = nn.Sequential(
            DepthwiseSeparableConv3d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # 编码路径
        self.enc_conv = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, ceil_mode=True)
        current_ch = out_ch
        self.enc_channels = []

        for i in range(height - 1):
            next_ch = mid_ch if i == 0 else min(mid_ch * (2 ** i), 32)  # 限制最大通道数
            self.enc_conv.append(nn.Sequential(
                DepthwiseSeparableConv3d(current_ch, next_ch, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            ))
            self.enc_channels.append(next_ch)
            current_ch = next_ch

        # 中间层
        self.mid_conv = nn.Sequential(
            DepthwiseSeparableConv3d(current_ch, current_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # 解码路径
        self.dec_conv = nn.ModuleList()
        reversed_enc_channels = list(reversed(self.enc_channels))

        for dec_ch in reversed_enc_channels:
            self.dec_conv.append(nn.Sequential(
                DepthwiseSeparableConv3d(dec_ch * 2, dec_ch, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            ))

        # 输出融合
        self.conv_out = nn.Sequential(
            DepthwiseSeparableConv3d(out_ch + reversed_enc_channels[0] if reversed_enc_channels else out_ch, out_ch, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # 残差连接（增强梯度流动）
        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            DepthwiseSeparableConv3d(in_ch, out_ch, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        identity = self.skip(x)  # 残差基准值
        x = self.conv_in(x)
        features = [x]

        # 编码（添加梯度检查点时确保输入可导）
        for conv in self.enc_conv:
            x = checkpoint(conv, x, use_reentrant=False)
            features.append(x)
            x = self.pool(x)

        # 中间层
        x = checkpoint(self.mid_conv, x, use_reentrant=False)

        # 解码
        for i, dec in enumerate(self.dec_conv):
            # 上采样到与编码特征同尺寸（使用align_corners=False避免数值波动）
            x = F.interpolate(x, size=features[-1].shape[2:], mode='trilinear', align_corners=False)
            enc_feature = features.pop()
            x = torch.cat([x, enc_feature], dim=1)  # 拼接特征
            x = checkpoint(dec, x, use_reentrant=False)

        # 融合+残差连接
        x = torch.cat([x, features[0]], dim=1)
        x = self.conv_out(x)
        return x + identity  # 残差输出（直接相加，避免激活后再相加导致梯度削弱）


class U2Net(nn.Module):
    """3D U2Net模型，增强整体稳定性"""

    def __init__(self, in_channel=1, out_channel=1, training=True):
        super().__init__()
        self.training = training

        # 编码器（逐步升维）
        self.enc1 = RSU3D(in_ch=in_channel, mid_ch=4, out_ch=8, height=2)
        self.pool1 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc2 = RSU3D(in_ch=8, mid_ch=8, out_ch=16, height=2)
        self.pool2 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc3 = RSU3D(in_ch=16, mid_ch=16, out_ch=32, height=2)
        self.pool3 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc4 = RSU3D(in_ch=32, mid_ch=32, out_ch=64, height=2)

        # 解码器（逐步降维）
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            DepthwiseSeparableConv3d(64, 32, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dec1 = RSU3D(in_ch=64, mid_ch=16, out_ch=32, height=2)  # 32（up1）+32（enc3）=64

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            DepthwiseSeparableConv3d(32, 16, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dec2 = RSU3D(in_ch=32, mid_ch=8, out_ch=16, height=2)  # 16+16=32

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            DepthwiseSeparableConv3d(16, 8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dec3 = RSU3D(in_ch=16, mid_ch=4, out_ch=8, height=2)  # 8+8=16

        # 最终输出层（单独初始化）
        self.final = nn.Conv3d(8, out_channel, 1)
        nn.init.kaiming_normal_(self.final.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.final.bias)

        # 多尺度输出分支（增强训练稳定性）
        self.side1 = self._side_branch(64, out_channel, 16)
        self.side2 = self._side_branch(32, out_channel, 8)
        self.side3 = self._side_branch(16, out_channel, 4)
        self.side4 = self._side_branch(8, out_channel, 2)

    def _side_branch(self, in_ch, out_ch, scale):
        """侧分支添加批归一化和激活"""
        return nn.Sequential(
            DepthwiseSeparableConv3d(in_ch, out_ch, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        # 确保输入可导（训练模式）
        if self.training and not x.requires_grad:
            x = x.requires_grad_(True)

        # 编码器前向
        e1 = checkpoint(self.enc1, x, use_reentrant=False)
        e2 = checkpoint(self.enc2, self.pool1(e1), use_reentrant=False)
        e3 = checkpoint(self.enc3, self.pool2(e2), use_reentrant=False)
        e4 = checkpoint(self.enc4, self.pool3(e3), use_reentrant=False)

        # 解码器前向
        up1_e4 = self.up1(e4)
        d1 = checkpoint(self.dec1, torch.cat([up1_e4, e3], dim=1), use_reentrant=False)

        up2_d1 = self.up2(d1)
        d2 = checkpoint(self.dec2, torch.cat([up2_d1, e2], dim=1), use_reentrant=False)

        up3_d2 = self.up3(d2)
        d3 = checkpoint(self.dec3, torch.cat([up3_d2, e1], dim=1), use_reentrant=False)

        final = self.final(d3)  # 最终输出（未激活，交给损失函数处理）

        if self.training:
            # 多尺度输出（均调整到输入尺寸）
            output1 = F.interpolate(self.side1(e4), size=x.shape[2:], mode='trilinear', align_corners=False)
            output2 = F.interpolate(self.side2(d1), size=x.shape[2:], mode='trilinear', align_corners=False)
            output3 = F.interpolate(self.side3(d2), size=x.shape[2:], mode='trilinear', align_corners=False)
            output4 = F.interpolate(self.side4(d3), size=x.shape[2:], mode='trilinear', align_corners=False)
            return output1, output2, output3, output4, final
        else:
            # 推理模式输出概率
            return torch.sigmoid(final)

