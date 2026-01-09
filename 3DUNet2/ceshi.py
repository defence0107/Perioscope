import torch
import torch.nn as nn
from models import VNet  # 假设您的模型保存在model.py文件中


def test_vnet_with_dummy_data():
    # 设置参数
    batch_size = 2
    in_channels = 1
    out_channels = 1
    depth, height, width = 64, 64, 64  # 3D体积数据尺寸

    # 创建虚拟输入数据
    dummy_input = torch.randn(batch_size, in_channels, depth, height, width)
    print(f"输入数据形状: {dummy_input.shape}")

    # 初始化模型(测试不同配置)
    configs = [
        {"training": True, "use_attention": True, "norm_type": "batch"},
        {"training": False, "use_attention": False, "norm_type": "instance"},
        {"training": True, "use_attention": True, "final_activation": False}
    ]

    for i, config in enumerate(configs):
        print(f"\n测试配置 {i + 1}: {config}")

        # 创建模型
        model = VNet.VNet(
            in_channel=in_channels,
            out_channel=out_channels,
            training=config.get("training", True),
            use_attention=config.get("use_attention", True),
            norm_type=config.get("norm_type", "batch"),
            final_activation=config.get("final_activation", True)
        )

        # 前向传播
        outputs = model(dummy_input)

        # 检查输出
        if config.get("training", True):
            # 训练模式下应返回5个输出(多尺度输出+主输出)
            assert len(outputs) == 5, "训练模式下应返回5个输出"
            for j, out in enumerate(outputs[:-1]):
                print(f"多尺度输出 {j + 1} 形状: {out.shape}")
                assert out.shape == (batch_size, out_channels, depth, height, width), f"多尺度输出 {j + 1} 形状不正确"

            main_output = outputs[-1]
            print(f"主输出形状: {main_output.shape}")
            assert main_output.shape == (batch_size, out_channels, depth, height, width), "主输出形状不正确"
        else:
            # 测试模式下只返回主输出
            print(f"测试模式输出形状: {outputs.shape}")
            assert outputs.shape == (batch_size, out_channels, depth, height, width), "测试模式输出形状不正确"

        # 检查激活函数
        if config.get("final_activation", True):
            assert torch.all(outputs[-1] >= 0) and torch.all(outputs[-1] <= 1), "输出值应在[0,1]范围内(Sigmoid激活)"
        else:
            assert torch.min(outputs[-1]) < 0 or torch.max(outputs[-1]) > 1, "输出值可能没有经过Sigmoid激活"

        print("测试通过!")


if __name__ == "__main__":
    test_vnet_with_dummy_data()