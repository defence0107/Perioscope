import pandas as pd
import os


def process_periodontitis_data(input_file, output_file=None, sheet_name='Sheet1', target_columns=None):
    """
    处理牙周炎数据Excel表格，将1级转换为0，2/3级转换为1

    参数:
    input_file (str): 输入Excel文件路径
    output_file (str): 输出Excel文件路径，默认为在输入文件名后添加"_processed"
    sheet_name (str): 要处理的表名
    target_columns (list): 要处理的目标列名列表，默认为None表示处理所有数值列
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 设置默认输出文件名
    if output_file is None:
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_processed{file_ext}"

    try:
        # 读取Excel文件
        df = pd.read_excel(input_file, sheet_name=sheet_name)

        # 转换函数：将1转为0，2和3转为1
        def convert_stage(value):
            # 尝试将值转换为数值类型
            try:
                num_value = float(value)
                if num_value == 1:
                    return 0
                elif num_value == 2:
                    return 1
                elif num_value == 3:
                    return 2
            except (ValueError, TypeError):
                pass  # 非数值保持原样
            return value

        # 确定要处理的列
        columns_to_process = []
        if target_columns:
            # 处理指定列
            columns_to_process = [col for col in target_columns if col in df.columns]
            if not columns_to_process:
                print(f"警告: 指定的列 {target_columns} 都不存在于工作表中")
        else:
            # 处理所有数值列
            columns_to_process = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if not columns_to_process:
                print("警告: 工作表中没有找到数值列")

        # 对确定的列应用转换函数
        for col in columns_to_process:
            df[col] = df[col].apply(convert_stage)
            # 转换为数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 写入处理后的Excel文件
        df.to_excel(output_file, sheet_name=sheet_name, index=False)

        print(f"处理完成! 已保存至: {output_file}")
        return output_file

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


# 示例用法
if __name__ == "__main__":
    input_file = "E:/D/periodontitis_data.xlsx"  # 替换为实际文件路径

    # 指定要处理的列名
    target_columns = ['Periodontitis']  # 替换为实际列名

    output_file = process_periodontitis_data(
        input_file,
        sheet_name='Sheet1',  # 确保与实际工作表名称一致
        target_columns=target_columns
    )