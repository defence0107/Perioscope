import torch


def calculate_bmi(weight, height):
    """
    计算BMI（身体质量指数）

    参数:
        weight (torch.Tensor): 体重张量（单位：千克）
        height (torch.Tensor): 身高张量（单位：米）

    返回:
        torch.Tensor: BMI值张量
    """
    return weight / (height ** 2)


def bmi_risk_level(bmi_tensor):
    """
    根据BMI值生成健康风险等级 (1-4级)

    参数:
        bmi_tensor (torch.Tensor): BMI值张量

    返回:
        torch.Tensor: 包含风险等级(1-4)的整数张量
    """
    levels = torch.full_like(bmi_tensor, 1, dtype=torch.long)
    levels = torch.where(bmi_tensor >= 18.5, torch.tensor(2, dtype=torch.long), levels)
    levels = torch.where(bmi_tensor >= 25.0, torch.tensor(3, dtype=torch.long), levels)
    levels = torch.where(bmi_tensor >= 30.0, torch.tensor(4, dtype=torch.long), levels)
    return levels


# 整合后的完整示例
if __name__ == "__main__":
    # 原始数据（支持批量计算）
    weight = torch.tensor(75)  # 体重（kg）
    height = torch.tensor([1.72])  # 身高（m）

    # 计算BMI
    bmi = calculate_bmi(weight, height)

    # 获取风险等级
    risk_levels = bmi_risk_level(bmi)

    # 完整结果输出
    print("体重(kg):", weight.numpy())
    print("身高(m): ", height.numpy())
    print("\nBMI计算结果:")
    print(bmi)
    print("\n健康风险等级:")
    print(risk_levels.numpy())

    # 风险等级说明
    print("\n[风险等级说明]")
    print("1: 体重不足 | 2: 正常范围 | 3: 超重 | 4: 肥胖")
    print("对应BMI范围:")
    print("<18.5    | 18.5-24.9 | 25.0-29.9 | ≥30.0")