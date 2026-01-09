import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


class ExcelDataset(Dataset):
    def __init__(self, excel_file, sheet_name=0, transform=None,
                 handle_outliers=True, normalize_features=False):
        # 读取Excel文件
        try:
            # 读取数据，不将任何行作为表头（保持原始数据）
            self.dataframe = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

            # 验证至少有两列数据（1列标签 + 至少1列特征）
            if self.dataframe.shape[1] < 2:
                raise ValueError(
                    f"Excel文件至少需要包含两列数据（1列标签 + 1列特征），但实际有{self.dataframe.shape[1]}列")

            # 数据预处理 - 移除空值
            self.dataframe.dropna(inplace=True)

            # 确保所有列都是数值类型
            for col in self.dataframe.columns:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')
            self.dataframe.dropna(inplace=True)

            # 验证数据集不为空
            if len(self.dataframe) == 0:
                raise ValueError("处理后的数据框为空，请检查Excel文件格式和数据内容")

            # 明确指定第一列（索引0）作为标签列
            label_column_index = 0
            self.label_column = self.dataframe.columns[label_column_index]
            label_values = self.dataframe[self.label_column].astype(int)  # 强制转换为整数

            # 验证标签值仅包含0、1、2
            valid_labels = {0, 1, 2}
            actual_labels = set(label_values.unique())
            if not actual_labels.issubset(valid_labels):
                invalid_labels = actual_labels - valid_labels
                raise ValueError(f"标签列（第一列）包含无效值: {invalid_labels}。必须只包含 0, 1, 2")

            # 确保标签列正确设置
            self.dataframe[self.label_column] = label_values

            # 处理异常值（仅处理特征列，即第一列之后的所有列）
            if handle_outliers:
                self._handle_outliers()

            # 特征标准化（仅标准化特征列）
            self.normalize = normalize_features
            self.scaler = None
            if self.normalize:
                self._normalize_features()

        except Exception as e:
            raise RuntimeError(f"处理Excel文件时出错: {str(e)}") from e

        self.transform = transform

    def _handle_outliers(self):
        """使用IQR方法处理特征列中的异常值（第一列之后的所有列）"""
        # 明确排除第一列（标签列），只处理特征列
        feature_columns = self.dataframe.columns[1:]

        for col in feature_columns:
            # 计算IQR
            q1 = self.dataframe[col].quantile(0.25)
            q3 = self.dataframe[col].quantile(0.75)
            iqr = q3 - q1

            # 确定异常值边界
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # 截断异常值
            self.dataframe[col] = self.dataframe[col].clip(lower_bound, upper_bound)

    def _normalize_features(self):
        """标准化特征列（第一列之后的所有列）"""
        # 明确排除第一列（标签列），只标准化特征列
        feature_columns = self.dataframe.columns[1:]

        self.scaler = StandardScaler()
        self.dataframe[feature_columns] = self.scaler.fit_transform(
            self.dataframe[feature_columns]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        # 明确提取第一列作为标签，其余作为特征
        features = torch.tensor(np.array(sample.iloc[1:], dtype=np.float32))
        label_value = int(sample.iloc[0])  # 第一列作为标签

        # 再次验证标签值有效性
        if label_value not in {0, 1, 2}:
            raise ValueError(f"无效的标签值 {label_value}，必须是0、1或2")

        label = torch.tensor(label_value, dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label


def create_dataloaders(excel_file, sheet_name=0, batch_size=32,
                       train_split=0.8, shuffle=True, transform=None,
                       handle_outliers=True, normalize_features=False):
    """
    创建训练和验证数据加载器，确保第一列作为标签列

    参数:
        excel_file: Excel文件路径
        sheet_name: 工作表名称或索引
        batch_size: 批次大小
        train_split: 训练集占比
        shuffle: 是否打乱数据
        transform: 特征变换函数
        handle_outliers: 是否处理异常值
        normalize_features: 是否标准化特征

    返回:
        训练数据加载器和验证数据加载器的元组
    """
    try:
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"文件不存在: {excel_file}")

        # 快速检查文件是否可读取，并验证至少有两列
        temp_df = pd.read_excel(excel_file, nrows=5, header=None)
        if temp_df.shape[1] < 2:
            raise ValueError(f"Excel文件至少需要包含两列数据（1列标签 + 1列特征），但实际有{temp_df.shape[1]}列")

        # 创建数据集
        dataset = ExcelDataset(
            excel_file,
            sheet_name=sheet_name,
            transform=transform,
            handle_outliers=handle_outliers,
            normalize_features=normalize_features
        )

        if len(dataset) == 0:
            raise ValueError("创建的数据集长度为0，请检查Excel文件内容")

        # 分割训练集和验证集
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    except Exception as e:
        raise RuntimeError(f"创建数据加载器失败: {str(e)}") from e


# 示例用法
if __name__ == "__main__":
    try:
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            "data.xlsx",
            batch_size=16,
            train_split=0.7,
            handle_outliers=True,
            normalize_features=True
        )

        print(f"训练集批次数量: {len(train_loader)}")
        print(f"验证集批次数量: {len(val_loader)}")

        # 查看一个批次的数据并验证标签
        for features, labels in train_loader:
            print(f"特征形状: {features.shape}")
            print(f"标签形状: {labels.shape}")
            print(f"标签示例: {labels[:5]}")
            # 验证标签值范围
            unique_labels = torch.unique(labels)
            print(f"批次中的唯一标签: {unique_labels}")
            assert torch.all((unique_labels >= 0) & (unique_labels <= 2)), "发现超出范围的标签值"
            break

    except Exception as e:
        print(f"发生错误: {e}")
