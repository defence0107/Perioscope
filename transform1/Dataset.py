import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ExcelDataset(Dataset):
    def __init__(self, excel_file, sheet_name=0, transform=None):
        # 读取Excel文件，忽略标题行
        self.dataframe = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

        # 数据清洗加强版
        self._clean_data()

        # 转换标签为四分类
        self._convert_labels()

        self.transform = transform

    def _clean_data(self):
        """强化数据清洗流程"""
        # 删除包含空值的行
        self.dataframe.dropna(inplace=True)

        # 确保所有列为数值类型
        for col in self.dataframe.columns:
            # 先转换为字符串处理特殊字符
            self.dataframe[col] = self.dataframe[col].astype(str)
            # 移除非数字字符（如逗号、百分号等）
            self.dataframe[col] = self.dataframe[col].str.replace(r'[^0-9\.-]', '', regex=True)
            # 转换为数值型
            self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')

        # 再次删除无效行
        self.dataframe.dropna(inplace=True)
        self.dataframe = self.dataframe.reset_index(drop=True)

    def _convert_labels(self):
        """安全转换标签为四分类"""
        # 确保首列为标签列
        label_col = self.dataframe.iloc[:, 0]

        # 打印原始标签分布
        print("\n原始标签分布:")
        print(label_col.value_counts())

        # 验证标签范围
        unique_labels = label_col.unique()
        print(f"唯一标签值: {unique_labels}")

        if len(unique_labels) < 4:
            print(f"警告: 发现{len(unique_labels)}个类别，少于预期的4个")
            print("继续处理，但可能导致分类器性能不佳")

        # 创建标签映射
        label_map = {val: i for i, val in enumerate(sorted(unique_labels))}
        print(f"标签映射: {label_map}")

        # 应用映射
        self.dataframe.iloc[:, 0] = label_col.map(label_map)

        # 验证转换后有4个类别
        final_labels = self.dataframe.iloc[:, 0].unique()
        print(f"转换后标签类别: {final_labels}")

        if len(final_labels) != 4:
            print("标签转换后未得到4个类别，实际得到:")
            print(self.dataframe.iloc[:, 0].value_counts())
            raise ValueError("标签转换后未得到有效四分类")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 修改特征提取维度
        features = self.dataframe.iloc[idx, 1:].values.astype(np.float32)  # 显式转换为float32
        label = self.dataframe.iloc[idx, 0].astype(np.int64)  # 确保为int64

        # 增加维度检查
        if features.shape[0] == 0:
            raise ValueError(f"样本{idx}特征为空")

        return torch.from_numpy(features), torch.tensor(label)


# 验证数据集
def test_dataset():
    excel_file = 'E:/D/shujubiao.xlsx'
    dataset = ExcelDataset(excel_file)

    # 打印样本信息
    print("\n数据集验证:")
    print(f"总样本数: {len(dataset)}")
    print(f"特征维度: {dataset[0][0].shape}")
    print(f"特征数据类型: {dataset[0][0].dtype}")
    print(f"标签分布:\n{dataset.dataframe.iloc[:, 0].value_counts()}")

    return dataset


# 示例使用
if __name__ == "__main__":
    dataset = test_dataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 验证DataLoader输出
    batch_features, batch_labels = next(iter(dataloader))
    print("\nDataLoader验证:")
    print(f"批量特征形状: {batch_features.shape}")
    print(f"特征数据类型: {batch_features.dtype}")
    print(f"标签数据类型: {batch_labels.dtype}")
    print(f"标签类别: {batch_labels.unique()}")

