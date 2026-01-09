import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer3D(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0):
        super(DenseLayer3D, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], 1)


class DenseBlock3D(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer3D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=3, in_channels=1):
        super(DenseNet3D, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Build dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition3D(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# Predefined 3D DenseNet models
def densenet121_3d(**kwargs):
    return DenseNet3D(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        **kwargs
    )


def densenet169_3d(**kwargs):
    return DenseNet3D(
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        bn_size=4,
        **kwargs
    )


def densenet201_3d(**kwargs):
    return DenseNet3D(
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        num_init_features=64,
        bn_size=4,
        **kwargs
    )


# Test code
if __name__ == '__main__':
    # Input example (batch=2, channel=1, depth=64, height=64, width=64)
    input_tensor = torch.randn(2, 1, 64, 64, 64)

    # Test DenseNet121-3D
    model = densenet121_3d(num_classes=3, in_channels=1)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Should be torch.Size([2, 3])

    # Parameter count
    print("Parameters:", sum(p.numel() for p in model.parameters()))