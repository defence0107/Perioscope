import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义文件路径
input_file = "E:/D/牙周炎.xlsx"
output_csv = "E:/D/牙周炎分级结果_全量.csv"
output_excel = "E:/D/牙周炎分级结果_完整版.xlsx"
output_chart = "E:/D/牙周炎影响因素分析.png"

# 创建输出目录
output_dir = os.path.dirname(output_excel)
os.makedirs(output_dir, exist_ok=True)

try:
    # 读取数据，排除第一列（假设第一列为periodontal）
    df = pd.read_excel(input_file).iloc[:, 1:]
    print(f"成功读取数据: {len(df)} 行，{len(df.columns)} 列")
    print("数据列名：", df.columns.tolist())

    # 检查必要列（确保包含分级所需字段：Smoke, Diabetes, BMI等相关因素字段 ）
    required_columns = ['Smoke', 'Diabetes', 'BMI']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要列: {missing_columns}")

    # 定义牙周炎分级函数，按照新分级标准
    def calculate_grading(row):
        smoke = row['Smoke']
        diabetes = row['Diabetes']
        bmi = row['BMI']
        # 宿主免疫反应正常判断（这里简单假设无糖尿病且不吸烟为免疫正常，实际需更多信息判断 ）
        immune_normal = not diabetes and not smoke
        # 危险因素数量统计
        risk_factors = 0
        if smoke:
            risk_factors += 1
        if diabetes:
            risk_factors += 1
        if bmi >= 28:
            risk_factors += 1
        # 牙周破坏加速史等信息假设无，实际可根据数据补充判断
        has_rapid_destruction_history = False

        if immune_normal and risk_factors == 0 and not has_rapid_destruction_history:
            return 'A级（低风险）'
        elif (not immune_normal or risk_factors > 0) and (not (diabetes and smoke > 10) and not has_rapid_destruction_history):
            return 'B级（中风险）'
        else:
            return 'C级（高风险）'

    # 应用分级函数
    df['牙周炎分级'] = df.apply(calculate_grading, axis=1)

    # 统计分级分布
    grade_distribution = df['牙周炎分级'].value_counts().reset_index()
    grade_distribution.columns = ['分级', '数量']
    print("\n分级结果分布：")
    print(grade_distribution)

    # 影响因素分析（排除所有分期相关列）
    def analyze_factors(data):
        categorical_vars = [
            "gender", "age", "weight", "Educational level", "Annual family income",
            "Drink", "Drinking frequency", "Smoke", "Smoking frequency",
            "Degree of smoking", "healthy diet", "Regular diet", "trouble sleeping",
            "sleep disorder", "Diabetes", "Cardiovascular disease",
            "Respiratory system diseases", "Rheumatoid arthritis", "Alzheimer's disease"
        ]
        continuous_vars = ["age", "weight", "BMI", "Annual family income"]

        results = []
        for var in categorical_vars:
            if var not in data.columns:
                print(f"跳过不存在的列: {var}")
                continue
            try:
                contingency_table = pd.crosstab(data[var], data['牙周炎分级'])
                chi2, p, _, _ = stats.chi2_contingency(contingency_table)
                results.append({
                    '变量': var,
                    '检验方法': '卡方检验',
                    '卡方值': chi2,
                    'p值': p,
                    '显著性': '显著' if p < 0.05 else '不显著'
                })
            except Exception as e:
                print(f"分析 {var} 时出错: {e}")

        for var in continuous_vars:
            if var not in data.columns:
                print(f"跳过不存在的列: {var}")
                continue
            try:
                groups = []
                for grade in data['牙周炎分级'].unique():
                    groups.append(data[data['牙周炎分级'] == grade][var].dropna())
                h, p = stats.kruskal(*groups)
                results.append({
                    '变量': var,
                    '检验方法': 'Kruskal-Wallis检验',
                    '统计量': h,
                    'p值': p,
                    '显著性': '显著' if p < 0.05 else '不显著'
                })
            except Exception as e:
                print(f"分析 {var} 时出错: {e}")

        results_df = pd.DataFrame(results).sort_values('p值')
        return results_df

    # 执行影响因素分析
    factors_analysis = analyze_factors(df)

    # 准备输出列
    output_columns = [col for col in df.columns if col != 'periodontal'] + ['牙周炎分级']
    # 输出所有带分级结果的数据到CSV
    df[output_columns].to_csv(output_csv, index=False, na_rep='nan')
    print(f"\n已保存 {len(output_columns)} 列数据到 CSV 文件: {output_csv}")

    # 保存结果到新Excel文件
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 保存原始数据和分级结果
        df.to_excel(writer, sheet_name='原始数据+分级', index=False)
        # 保存分级分布
        grade_distribution.to_excel(writer, sheet_name='分级分布', index=False)
        # 保存影响因素分析结果
        factors_analysis.to_excel(writer, sheet_name='影响因素分析', index=False)

    print(f"Excel 文件保存成功: {output_excel}")

    # 可视化分析结果（仅显示显著因素）
    significant_factors = factors_analysis[factors_analysis['显著性'] == '显著']
    if not significant_factors.empty:
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            x='变量',
            y='-log(p值)',
            data=significant_factors.assign(**{'-log(p值)': -np.log10(significant_factors['p值'])}),
            palette='viridis'
        )
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')
        plt.title('各因素对牙周炎分级的影响显著性', fontsize=15)
        plt.xlabel('影响因素', fontsize=12)
        plt.ylabel('-log(p值)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_chart, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"影响因素可视化图表已保存: {output_chart}")
    else:
        print("没有找到显著影响因素，未生成可视化图表。")

    print("\n分析完成！")

except Exception as e:
    print(f"\n执行过程中发生错误: {e}")