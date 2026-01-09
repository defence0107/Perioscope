import pandas as pd

# 定义诊断函数
def diagnose_glucose(glucose):
    """根据WHO标准进行糖尿病诊断"""
    try:
        glucose = float(glucose)
        if glucose < 5.55:
            return '1'
        elif 5.55 <= glucose < 6.99:
            return '1'
        else:
            return '2'
    except:
        return '数据异常'

# 读取Excel文件（请修改为你的文件路径）
input_path = "E:/D/血糖数据.xlsx"
output_path = "E:/D/诊断结果.xlsx"

# 读取数据
df = pd.read_excel(input_path)

# 检查是否存在空腹血糖列
if 'Fasting Glucose (mg/dL)' not in df.columns:
    raise ValueError("Excel文件中缺少'Fasting Glucose (mg/dL)'列")

# 添加诊断列
df['诊断结果'] = df['Fasting Glucose (mg/dL)'].apply(diagnose_glucose)

# 保存结果
df.to_excel(output_path, index=False, engine='openpyxl')

print(f"诊断完成，结果已保存至：{output_path}")