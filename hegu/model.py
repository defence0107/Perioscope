import torch.nn as nn
import torch


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock3D, self).__init__()
        # 3D卷积核调整为3x3x3
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck3D, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        # 所有2D卷积替换为3D卷积
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)  # 3D padding保持各维度相同
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=3,
                 include_top=True,
                 in_channels=1,  # 医学影像通常单通道
                 groups=1,
                 width_per_group=64):
        super(ResNet3D, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        # 初始卷积层调整为3D
        self.conv1 = nn.Conv3d(in_channels, self.in_channel, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 特征层
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D全局平均池化
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel,
                            stride=stride, downsample=downsample,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入维度: (batch, channel, depth, height, width)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


# 预定义3D模型
def resnet18_3d(**kwargs):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)


def resnet34_3d(**kwargs):
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], **kwargs)


def resnet50_3d(**kwargs):
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)


def resnet101_3d(**kwargs):
    return ResNet3D(Bottleneck3D, [3, 4, 23, 3], **kwargs)


# 测试代码
if __name__ == '__main__':
    # 输入示例 (batch=2, channel=1, depth=64, height=64, width=64)
    input_tensor = torch.randn(2, 1, 64, 64, 64)

    # 测试ResNet50-3D
    model = resnet50_3d(num_classes=3, in_channels=1)
    output = model(input_tensor)
    print(f"输出维度: {output.shape}")  # 应为 torch.Size([2, 3])

    # 参数量统计
    print("参数量:", sum(p.numel() for p in model.parameters()))