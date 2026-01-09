import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class ChannelAttention(nn.Module):
    """通道注意力模块，可配置缩减比例"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块，可配置卷积核大小"""
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class MixedAttention(nn.Module):
    """混合注意力模块，结合通道和空间注意力"""
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(MixedAttention, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.channel_att(x)
        x = x * ca
        sa = self.spatial_att(x)
        x = x * sa
        return x


class ResidualBlock(nn.Module):
    """残差块，可选注意力机制和随机深度"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_attention: bool = True, drop_path_rate: float = 0.0,
                 norm_type: str = 'batch'):
        super(ResidualBlock, self).__init__()
        self.use_attention = use_attention
        self.drop_path_rate = drop_path_rate

        if use_attention:
            self.attention = MixedAttention(out_channels)

        # 归一化层工厂函数
        norm_layer = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        mask = torch.empty((x.shape[0], 1, 1, 1, 1), device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.use_attention:
            out = self.attention(out)

        if self.drop_path_rate > 0:
            out = self.drop_path(out)

        out += self.shortcut(identity)
        return F.relu(out)


class ConvGatedUnit(nn.Module):
    """卷积门控单元，可选注意力机制"""
    def __init__(self, in_channels: int, out_channels: int,
                 use_attention: bool = True, norm_type: str = 'batch'):
        super(ConvGatedUnit, self).__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = MixedAttention(out_channels)

        norm_layer = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d

        self.conv = nn.Conv3d(in_channels, out_channels * 2, kernel_size=3, padding=1, bias=False)
        self.bn = norm_layer(out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        value, gate = x.chunk(2, dim=1)
        if self.use_attention:
            value = self.attention(value)
        return value * torch.sigmoid(gate)


class VNet(nn.Module):
    """增强型3D V-Net架构，输出通道修改为2"""
    def __init__(self, in_channel: int = 1, out_channel: int = 2, training: bool = True,
                 use_attention: bool = True, drop_path_rate: float = 0.1,
                 init_channels: int = 16, reduction_ratio: int = 16,
                 final_activation: bool = True, norm_type: str = 'batch'):
        super(VNet, self).__init__()
        self.training_mode = training  # 重命名避免与nn.Module的training属性冲突
        self.final_activation = final_activation

        # 初始化权重
        self.apply(self._init_weights)

        # ---------------------------
        # 编码器部分 (Encoder)
        # ---------------------------
        self.encoder1 = ResidualBlock(
            in_channel, init_channels,
            drop_path_rate=drop_path_rate * 0.2,
            norm_type=norm_type
        )
        self.encoder2 = ResidualBlock(
            init_channels, init_channels * 2, stride=2,
            drop_path_rate=drop_path_rate * 0.4,
            norm_type=norm_type
        )
        self.encoder3 = ResidualBlock(
            init_channels * 2, init_channels * 4, stride=2,
            drop_path_rate=drop_path_rate * 0.6,
            norm_type=norm_type
        )
        self.encoder4 = ResidualBlock(
            init_channels * 4, init_channels * 8, stride=2,
            drop_path_rate=drop_path_rate * 0.8,
            norm_type=norm_type
        )

        # ---------------------------
        # 上采样部分 (Upsampling)
        # ---------------------------
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(init_channels * 4, init_channels * 4, kernel_size=1)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(init_channels * 2, init_channels * 2, kernel_size=1)
        )
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(init_channels, init_channels, kernel_size=1)
        )

        # ---------------------------
        # 特征融合部分 (Feature Fusion)
        # ---------------------------
        self.fusion1 = nn.Sequential(
            nn.Conv3d(init_channels + init_channels * 2, init_channels, kernel_size=1),
            nn.BatchNorm3d(init_channels) if norm_type == 'batch' else nn.InstanceNorm3d(init_channels),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv3d(init_channels * 2 + init_channels * 4, init_channels * 2, kernel_size=1),
            nn.BatchNorm3d(init_channels * 2) if norm_type == 'batch' else nn.InstanceNorm3d(init_channels * 2),
            nn.ReLU()
        )
        self.fusion3 = nn.Sequential(
            nn.Conv3d(init_channels * 4 + init_channels * 8, init_channels * 4, kernel_size=1),
            nn.BatchNorm3d(init_channels * 4) if norm_type == 'batch' else nn.InstanceNorm3d(init_channels * 4),
            nn.ReLU()
        )

        # ---------------------------
        # 解码器部分 (Decoder)
        # ---------------------------
        self.decoder3 = ConvGatedUnit(
            init_channels * 8, init_channels * 4,
            use_attention=use_attention,
            norm_type=norm_type
        )
        self.decoder2 = ConvGatedUnit(
            init_channels * 4 + init_channels * 4, init_channels * 2,
            use_attention=use_attention,
            norm_type=norm_type
        )
        self.decoder1 = ConvGatedUnit(
            init_channels * 2 + init_channels * 2, init_channels,
            use_attention=use_attention,
            norm_type=norm_type
        )

        # ---------------------------
        # 输出层 (Output Layers) - 修改为2通道输出
        # ---------------------------
        self.out_conv = nn.Sequential(
            nn.Conv3d(init_channels, out_channel, kernel_size=1),
            # 对于2通道输出，使用Softmax激活更合适（如多分类分割任务）
            nn.Softmax(dim=1) if final_activation else nn.Identity()
        )

        # ---------------------------
        # 多尺度输出层 (固定存在，修改为2通道输出)
        # ---------------------------
        self.map4 = nn.Sequential(
            nn.Conv3d(init_channels * 8, out_channel, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
        )
        self.map3 = nn.Sequential(
            nn.Conv3d(init_channels * 4, out_channel, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        )
        self.map2 = nn.Sequential(
            nn.Conv3d(init_channels * 2, out_channel, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )
        self.map1 = nn.Conv3d(init_channels, out_channel, kernel_size=1)

    def _init_weights(self, m: nn.Module):
        """初始化权重"""
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 编码器前向传播
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 特征融合
        e2_up = F.interpolate(e2, size=e1.shape[2:], mode='trilinear', align_corners=False)
        e1_fused = self.fusion1(torch.cat([e1, e2_up], dim=1))

        e3_up = F.interpolate(e3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        e2_fused = self.fusion2(torch.cat([e2, e3_up], dim=1))

        e4_up = F.interpolate(e4, size=e3.shape[2:], mode='trilinear', align_corners=False)
        e3_fused = self.fusion3(torch.cat([e3, e4_up], dim=1))

        # 解码器前向传播
        d3 = self.decoder3(e4)
        up3 = self.upsample3(d3)
        d2_input = torch.cat([up3, e3_fused], dim=1)

        d2 = self.decoder2(d2_input)
        up2 = self.upsample2(d2)
        d1_input = torch.cat([up2, e2_fused], dim=1)

        d1 = self.decoder1(d1_input)
        up1 = self.upsample1(d1)

        # 主输出
        output = self.out_conv(up1)

        # 训练时返回多尺度输出，测试时仅返回主输出
        if self.training_mode:
            return (
                self.map1(up1),
                self.map2(up2),
                self.map3(up3),
                self.map4(e4),
                output  # 主输出放在最后
            )
        return output