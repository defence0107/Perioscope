import torch


def check_model_weights(weight_path):
    try:
        # 加载权重文件（不加载到GPU，避免设备问题）
        state_dict = torch.load(weight_path, map_location='cpu')

        # 打印权重文件基本信息
        print(f"权重文件加载成功！类型：{type(state_dict)}")
        print(f"包含参数数量：{len(state_dict)}")
        print("\n前5个参数名称：")
        for i, (key, value) in enumerate(state_dict.items()):
            if i < 5:
                print(f"- {key} (形状：{value.shape}, 类型：{value.dtype})")
            else:
                break

        # 检查是否有异常值（如全0、全NaN）
        has_invalid = False
        for key, value in state_dict.items():
            if torch.isnan(value).any() or torch.all(value == 0):
                print(f"警告：参数 {key} 包含全0或NaN值！")
                has_invalid = True
        if not has_invalid:
            print("参数值未发现异常。")

        return True

    except Exception as e:
        print(f"加载失败！错误信息：{e}")
        return False


# 使用示例
weight_path = "D:/project/3DUNet2/saved_models/epoch_1500_checkpoint.pth"
is_valid = check_model_weights(weight_path)
if is_valid:
    print("模型权重文件完整且可用！")
else:
    print("模型权重文件可能损坏或格式错误。")
