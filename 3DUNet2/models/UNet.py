import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)

        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Encoder
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out  # Size after 1st pooling
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out  # Size after 2nd pooling
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out  # Size after 3rd pooling
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))

        # Decoder with cropping
        output1 = self.map1(out)

        # Decoder block 1
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='trilinear', align_corners=False))
        out = self._center_crop_or_pad(out, t3)
        out = torch.add(out, t3)
        output2 = self.map2(out)

        # Decoder block 2
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='trilinear', align_corners=False))
        out = self._center_crop_or_pad(out, t2)
        out = torch.add(out, t2)
        output3 = self.map3(out)

        # Decoder block 3
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='trilinear', align_corners=False))
        out = self._center_crop_or_pad(out, t1)
        out = torch.add(out, t1)

        # Final decoder step
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='trilinear', align_corners=False))
        output4 = self.map4(out)

        if self.training:
            return output1, output2, output3, output4
        else:
            return output4

    def _center_crop_or_pad(self, x, target):
        """
        中心裁剪或填充x以匹配target的尺寸

        参数:
        x: 需要调整的张量
        target: 目标尺寸的张量

        返回:
        调整后的x，尺寸与target相同
        """
        # 获取目标尺寸
        target_d, target_h, target_w = target.size(2), target.size(3), target.size(4)
        # 获取当前张量尺寸
        x_d, x_h, x_w = x.size(2), x.size(3), x.size(4)

        # 计算差值
        d_diff = x_d - target_d
        h_diff = x_h - target_h
        w_diff = x_w - target_w

        # 如果x比target大，进行裁剪
        if d_diff > 0 or h_diff > 0 or w_diff > 0:
            # 计算裁剪量
            d_start = d_diff // 2
            h_start = h_diff // 2
            w_start = w_diff // 2

            # 中心裁剪
            x = x[:, :,
                d_start:d_start + target_d if d_diff > 0 else slice(None),
                h_start:h_start + target_h if h_diff > 0 else slice(None),
                w_start:w_start + target_w if w_diff > 0 else slice(None)]

        # 如果x比target小，进行填充
        elif d_diff < 0 or h_diff < 0 or w_diff < 0:
            # 计算填充量
            d_pad_before = abs(d_diff) // 2
            d_pad_after = abs(d_diff) - d_pad_before
            h_pad_before = abs(h_diff) // 2
            h_pad_after = abs(h_diff) - h_pad_before
            w_pad_before = abs(w_diff) // 2
            w_pad_after = abs(w_diff) - w_pad_before

            # 使用零填充
            x = F.pad(x, (w_pad_before, w_pad_after, h_pad_before, h_pad_after, d_pad_before, d_pad_after))

        return x