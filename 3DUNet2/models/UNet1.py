import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True, deep_supervision=True):
        super(UNetPlusPlus, self).__init__()

        self.training = training
        self.deep_supervision = deep_supervision

        # 编码器部分 (保持原始结构)
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)

        # UNet++特有的嵌套解码器部分
        # 第1级解码器 (与原始UNet类似)
        self.decoder2_1 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # 对应原始decoder2
        self.decoder3_1 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # 对应原始decoder3
        self.decoder4_1 = nn.Conv3d(64, 32, 3, stride=1, padding=1)  # 对应原始decoder4

        # 第2级解码器 (增加的嵌套连接)
        self.decoder1_2 = nn.Conv3d(32 + 64, 32, 3, stride=1, padding=1)
        self.decoder2_2 = nn.Conv3d(128 + 128, 128, 3, stride=1, padding=1)
        self.decoder3_2 = nn.Conv3d(64 + 64, 64, 3, stride=1, padding=1)

        # 第3级解码器 (增加的嵌套连接)
        self.decoder1_3 = nn.Conv3d(32 + 32 + 64, 32, 3, stride=1, padding=1)
        self.decoder2_3 = nn.Conv3d(128 + 128 + 64, 128, 3, stride=1, padding=1)

        # 第4级解码器 (增加的嵌套连接)
        self.decoder1_4 = nn.Conv3d(32 + 32 + 32 + 64, 32, 3, stride=1, padding=1)

        # 输出映射层 (保持原始结构)
        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
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

        # 新增的深度监督输出层
        self.map4_2 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map3_2 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map2_2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 编码器路径 (保持原始结构)
        out_e1 = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))  # 32 channels
        t1 = out_e1  # Size after 1st pooling

        out_e2 = F.relu(F.max_pool3d(self.encoder2(out_e1), 2, 2))  # 64 channels
        t2 = out_e2  # Size after 2nd pooling

        out_e3 = F.relu(F.max_pool3d(self.encoder3(out_e2), 2, 2))  # 128 channels
        t3 = out_e3  # Size after 3rd pooling

        out_e4 = F.relu(F.max_pool3d(self.encoder4(out_e3), 2, 2))  # 256 channels

        # 第1级解码器 (与原始UNet类似)
        # 解码器块1
        out_d2_1 = F.relu(F.interpolate(self.decoder2_1(out_e4), scale_factor=2, mode='trilinear', align_corners=False))
        out_d2_1 = self._center_crop_or_pad(out_d2_1, t3)
        out_d2_1 = torch.add(out_d2_1, t3)  # 128 channels
        output2_1 = self.map2(out_d2_1)

        # 解码器块2
        out_d3_1 = F.relu(
            F.interpolate(self.decoder3_1(out_d2_1), scale_factor=2, mode='trilinear', align_corners=False))
        out_d3_1 = self._center_crop_or_pad(out_d3_1, t2)
        out_d3_1 = torch.add(out_d3_1, t2)  # 64 channels
        output3_1 = self.map3(out_d3_1)

        # 解码器块3
        out_d4_1 = F.relu(
            F.interpolate(self.decoder4_1(out_d3_1), scale_factor=2, mode='trilinear', align_corners=False))
        out_d4_1 = self._center_crop_or_pad(out_d4_1, t1)
        out_d4_1 = torch.add(out_d4_1, t1)  # 32 channels
        output4_1 = self.map4(out_d4_1)

        # 第2级解码器 (UNet++新增的嵌套连接)
        # 解码器块1_2
        out_d1_2 = torch.cat(
            [out_d4_1, F.relu(F.interpolate(out_d3_1, scale_factor=2, mode='trilinear', align_corners=False))], dim=1)
        out_d1_2 = self.decoder1_2(out_d1_2)
        output4_2 = self.map4_2(out_d1_2)

        # 解码器块2_2
        out_d2_2 = torch.cat(
            [out_d2_1, F.relu(F.interpolate(out_e4, scale_factor=2, mode='trilinear', align_corners=False))], dim=1)
        out_d2_2 = self.decoder2_2(out_d2_2)
        output2_2 = self.map2_2(out_d2_2)

        # 解码器块3_2
        out_d3_2 = torch.cat(
            [out_d3_1, F.relu(F.interpolate(out_d2_1, scale_factor=2, mode='trilinear', align_corners=False))], dim=1)
        out_d3_2 = self.decoder3_2(out_d3_2)
        output3_2 = self.map3_2(out_d3_2)

        # 第3级解码器 (UNet++新增的嵌套连接)
        # 解码器块1_3
        out_d1_3 = torch.cat([
            out_d4_1,
            F.relu(F.interpolate(out_d3_1, scale_factor=2, mode='trilinear', align_corners=False)),
            F.relu(F.interpolate(out_d3_2, scale_factor=4, mode='trilinear', align_corners=False))
        ], dim=1)
        out_d1_3 = self.decoder1_3(out_d1_3)

        # 第4级解码器 (UNet++新增的嵌套连接)
        # 解码器块1_4
        out_d1_4 = torch.cat([
            out_d4_1,
            F.relu(F.interpolate(out_d3_1, scale_factor=2, mode='trilinear', align_corners=False)),
            F.relu(F.interpolate(out_d3_2, scale_factor=4, mode='trilinear', align_corners=False)),
            F.relu(F.interpolate(out_d1_3, scale_factor=2, mode='trilinear', align_corners=False))
        ], dim=1)
        out_d1_4 = self.decoder1_4(out_d1_4)

        # 最终输出
        output1 = self.map1(out_e4)

        if self.training and self.deep_supervision:
            # 训练阶段且使用深度监督时返回所有输出
            return output1, output2_1, output2_2, output3_1, output3_2, output4_1, output4_2
        else:
            # 推理阶段或不使用深度监督时返回最终输出
            return output4_1

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