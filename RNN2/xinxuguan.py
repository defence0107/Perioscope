import pandas as pd

def check_cvd_risk(tc, ldlc, hdlc, tg):
    """单条数据风险评估函数"""
    # 指标分类标准（mg/dL）
    tc_level = ('理想' if tc < 5.172 else
                '边缘升高' if 5.172 <= tc < 6.180 else
                '升高')

    ldlc_level = ('理想' if ldlc < 2.586 else
                  '合适' if ldlc < 3.336 else
                  '边缘升高' if 3.336 <= ldlc < 4.112 else
                  '升高')

    hdlc_level = '降低' if hdlc < 1.293 else '正常'

    tg_level = ('合适' if tg < 1.694 else
                '边缘升高' if 1.694 <= tg < 2.247 else
                '升高')

    # 风险判断（只要有任何一项指标异常即判定为有风险）
    has_risk = (
        ldlc_level in ['升高', '边缘升高'] or
        hdlc_level == '降低' or
        tg_level in ['升高', '边缘升高'] or
        tc_level in ['升高', '边缘升高']
    )

    return {
        'TC_分类': f"{tc_level} ({tc} mg/dL)",
        'LDL_分类': f"{ldlc_level} ({ldlc} mg/dL)",
        'HDL_分类': f"{hdlc_level} ({hdlc} mg/dL)",
        'TG_分类': f"{tg_level} ({tg} mg/dL)",
        '心血管病风险评估': 2 if has_risk else 1
    }

def process_excel(input_path, output_path):
    """处理Excel主函数"""
    # 读取数据
    df = pd.read_excel(input_path, engine='openpyxl')

    # 处理每一行数据
    results = df.apply(lambda row: pd.Series(check_cvd_risk(
        tc=row['Total Cholesterol( mg/dL)'],
        ldlc=row['LDL(mg/dL)'],
        hdlc=row['HDL(mg/dL)'],
        tg=row['Triglyceride (mg/dL)']
    )), axis=1)

    # 合并结果
    output_df = pd.concat([df, results], axis=1)

    # 保存结果
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        output_df.to_excel(writer, index=False)
    print(f"处理完成，结果已保存至：{output_path}")

if __name__ == "__main__":
    # 使用示例
    input_file = "E:/D/血脂数据.xlsx"
    output_file = "E:/D/风险评估结果.xlsx"

    # 执行处理
    process_excel(input_file, output_file)