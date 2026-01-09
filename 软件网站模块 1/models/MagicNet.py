import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union


class ChannelAttention(nn.Module):
    """轻量级通道注意力模块"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 使用1x1x1卷积替代全连接层，减少参数
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, max(4, in_channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(4, in_channels // reduction), in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """轻量级空间注意力模块"""
    def __init__(self, kernel_size=3):  # 减小卷积核尺寸，降低计算量
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class MixedAttention(nn.Module):
    """混合注意力模块，支持可选启用"""
    def __init__(self, in_channels, reduction=8, kernel_size=3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.channel_att = ChannelAttention(in_channels, reduction)
            self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        if not self.use_attention:
            return x
        x = x * self.channel_att(x)
        return x * self.spatial_att(x)


class ResidualBlock(nn.Module):
    """带梯度检查点的残差块"""
    def __init__(self, in_channels, out_channels, stride=1,
                 use_attention=True, drop_path_rate=0.0,
                 norm_type='batch', use_checkpoint=False):
        super().__init__()
        self.use_attention = use_attention
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint  # 新增：是否启用梯度检查点

        norm_layer = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels)
        )

        if use_attention:
            self.attention = MixedAttention(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        # 定义检查点计算函数
        def _forward_blocks(x):
            x = self.conv1(x)
            x = self.conv2(x)
            if self.use_attention and hasattr(self, 'attention'):
                x = self.attention(x)
            return x

        identity = x

        if self.use_checkpoint and self.training:
            # 原代码：x = checkpoint(_forward_blocks, x)
            x = checkpoint(_forward_blocks, x, use_reentrant=False)
        else:
            x = _forward_blocks(x)

        # 改进的drop path实现，减少内存占用
        if self.drop_path_rate > 0 and self.training:
            keep_prob = 1 - self.drop_path_rate
            mask = torch.empty((x.shape[0], 1, 1, 1, 1), device=x.device, dtype=x.dtype)
            mask.bernoulli_(keep_prob)
            x = x * mask / keep_prob

        return F.relu(x + self.shortcut(identity))


class ConvGatedUnit(nn.Module):
    """优化的门控卷积单元"""
    def __init__(self, in_channels, out_channels,
                 use_attention=True, norm_type='batch',
                 use_checkpoint=False):
        super().__init__()
        self.use_attention = use_attention
        self.use_checkpoint = use_checkpoint

        norm_layer = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d

        self.conv = nn.Conv3d(in_channels, out_channels * 2, 3, padding=1, bias=False)
        self.bn = norm_layer(out_channels * 2)

        if use_attention:
            self.attention = MixedAttention(out_channels)

    def forward(self, x):
        def _forward(x):
            x = self.bn(self.conv(x))
            value, gate = x.chunk(2, dim=1)
            if self.use_attention and hasattr(self, 'attention'):
                value = self.attention(value)
            return value * torch.sigmoid(gate)

        if self.use_checkpoint and self.training:
            # 原代码：return checkpoint(_forward, x)
            return checkpoint(_forward, x, use_reentrant=False)  # 添加参数
        else:
            return _forward(x)


class MagicNet(nn.Module):
    """优化后的MagicNet模型，减少显存占用"""
    def __init__(self, in_channel=1, out_channel=1, training=True,
                 use_attention=True, drop_path_rate=0.1,
                 init_channels=8, reduction_ratio=8,
                 final_activation=True, norm_type='batch',
                 use_dense_connections=True,
                 use_checkpoint=True,  # 新增：启用梯度检查点
                 light_weight=False):  # 新增：轻量化模式
        super().__init__()
        self.training = training
        self.final_activation = final_activation
        self.use_dense_connections = use_dense_connections
        self.use_checkpoint = use_checkpoint

        # 轻量化模式：减少通道数和网络深度
        if light_weight:
            init_channels = max(4, init_channels // 2)
            reduction_ratio = min(16, reduction_ratio * 2)

        # 编码器：添加梯度检查点支持
        self.encoder1 = ResidualBlock(
            in_channel, init_channels,
            drop_path_rate=drop_path_rate * 0.2,
            norm_type=norm_type,
            use_attention=use_attention,
            use_checkpoint=use_checkpoint
        )
        self.encoder2 = ResidualBlock(
            init_channels, init_channels * 2, stride=2,
            drop_path_rate=drop_path_rate * 0.4,
            norm_type=norm_type,
            use_attention=use_attention,
            use_checkpoint=use_checkpoint
        )
        self.encoder3 = ResidualBlock(
            init_channels * 2, init_channels * 4, stride=2,
            drop_path_rate=drop_path_rate * 0.6,
            norm_type=norm_type,
            use_attention=use_attention,
            use_checkpoint=use_checkpoint
        )
        # 轻量化模式下减少最深层的通道数
        encoder4_out = init_channels * 8 if not light_weight else init_channels * 6
        self.encoder4 = ResidualBlock(
            init_channels * 4, encoder4_out, stride=2,
            drop_path_rate=drop_path_rate * 0.8,
            norm_type=norm_type,
            use_attention=use_attention,
            use_checkpoint=use_checkpoint
        )

        # 简化瓶颈层
        self.bottleneck = nn.Sequential(
            ResidualBlock(
                encoder4_out, encoder4_out,
                drop_path_rate=drop_path_rate,
                norm_type=norm_type,
                use_attention=use_attention,
                use_checkpoint=use_checkpoint
            ),
            nn.Conv3d(encoder4_out, encoder4_out, 1),
            nn.BatchNorm3d(encoder4_out) if norm_type == 'batch' else nn.InstanceNorm3d(encoder4_out),
            nn.ReLU()
        )

        # 解码器
        self.decoder4 = self._make_decoder_block(
            encoder4_out * 2, init_channels * 4,
            use_attention=use_attention,
            norm_type=norm_type,
            use_checkpoint=use_checkpoint
        )
        self.decoder3 = self._make_decoder_block(
            init_channels * 4 + (init_channels * 4 if use_dense_connections else 0),
            init_channels * 2,
            use_attention=use_attention,
            norm_type=norm_type,
            use_checkpoint=use_checkpoint
        )
        self.decoder2 = self._make_decoder_block(
            init_channels * 2 + (init_channels * 2 if use_dense_connections else 0),
            init_channels,
            use_attention=use_attention,
            norm_type=norm_type,
            use_checkpoint=use_checkpoint
        )
        self.decoder1 = self._make_decoder_block(
            init_channels + (init_channels if use_dense_connections else 0),
            init_channels,
            use_attention=use_attention,
            norm_type=norm_type,
            use_checkpoint=use_checkpoint
        )

        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv3d(init_channels, out_channel, 1),
            nn.Sigmoid() if final_activation else nn.Identity()
        )

        # 多尺度输出（优化存储）
        if self.training:
            self.map4 = nn.Sequential(
                nn.Conv3d(init_channels * 4, out_channel, 1),
                nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
            )
            self.map3 = nn.Sequential(
                nn.Conv3d(init_channels * 2, out_channel, 1),
                nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
            )
            self.map2 = nn.Sequential(
                nn.Conv3d(init_channels, out_channel, 1),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            )
            self.map1 = nn.Conv3d(init_channels, out_channel, 1)

        self._init_weights()

    def _make_decoder_block(self, in_channels, out_channels, use_attention, norm_type, use_checkpoint):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvGatedUnit(
                in_channels, out_channels,
                use_attention=use_attention,
                norm_type=norm_type,
                use_checkpoint=use_checkpoint
            ),
            ConvGatedUnit(
                out_channels, out_channels,
                use_attention=use_attention,
                norm_type=norm_type,
                use_checkpoint=use_checkpoint
            )
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码器路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 瓶颈层
        b = self.bottleneck(e4)

        # 解码器路径（优化拼接操作的内存使用）
        d4 = self.decoder4(torch.cat([b, e4], dim=1))

        # 条件拼接，减少不必要的内存占用
        d3_input = torch.cat([d4, e3], dim=1) if self.use_dense_connections else d4
        d3 = self.decoder3(d3_input)

        d2_input = torch.cat([d3, e2], dim=1) if self.use_dense_connections else d3
        d2 = self.decoder2(d2_input)

        d1_input = torch.cat([d2, e1], dim=1) if self.use_dense_connections else d2
        d1 = self.decoder1(d1_input)

        # 最终输出
        output = self.out_conv(d1)

        if self.training:
            # 训练时返回多尺度输出（按需计算，减少同时存储）
            return (self.map1(d1),
                    self.map2(d2),
                    self.map3(d3),
                    self.map4(d4),
                    output)
        return output


# 混合精度训练工具类
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, data, target):
        self.model.train()
        self.optimizer.zero_grad()

        data, target = data.to(self.device), target.to(self.device)

        # 混合精度前向传播
        with torch.cuda.amp.autocast():
            outputs = self.model(data)
            # 计算多尺度损失
            if isinstance(outputs, tuple):
                loss = sum(self.criterion(out, target) for out in outputs) / len(outputs)
            else:
                loss = self.criterion(outputs, target)

        # 混合精度反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def validate(self, data, target):
        self.model.eval()
        with torch.no_grad():
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]  # 取最终输出
            loss = self.criterion(outputs, target).item()
        return loss, outputs
