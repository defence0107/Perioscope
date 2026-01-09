import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


class ExcelDataset(Dataset):
    """
    用于读取Excel数据的数据集类，支持四分类任务的数据预处理

    参数:
        excel_file: Excel文件路径
        sheet_name: 工作表名称或索引
        transform: 特征转换函数
        preprocessing: 预处理选项字典
            - normalize: 是否标准化特征 (bool)
            - balance_classes: 是否平衡类别 (bool)
            - balance_method: 类别平衡方法 ('oversample', 'undersample')
            - test_size: 测试集比例 (float)
            - random_state: 随机种子 (int)
    """

    def __init__(self, excel_file, sheet_name=0, transform=None,
                 preprocessing=None):
        # 设置默认预处理参数
        self.preprocessing = {
            'normalize': True,
            'balance_classes': True,
            'balance_method': 'oversample',  # 或'undersample'
            'test_size': 0.2,
            'random_state': 42,
            **(preprocessing or {})
        }

        # 读取Excel文件
        self._load_data(excel_file, sheet_name)

        # 标准化特征
        if self.preprocessing['normalize']:
            self._normalize_features()

        # 平衡类别
        if self.preprocessing['balance_classes']:
            self._balance_classes()

        self.transform = transform

    def _load_data(self, excel_file, sheet_name):
        """加载并验证Excel数据"""
        try:
            # 读取数据
            self.dataframe = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

            # 基本预处理
            self.dataframe.dropna(inplace=True)

            # 确保所有列都是数值类型
            for col in self.dataframe.columns:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')
            self.dataframe.dropna(inplace=True)

            # 验证标签列是否只包含0-3的整数
            if len(self.dataframe) > 0:
                label_column = self.dataframe.iloc[:, 0]
                if not set(label_column.unique()).issubset({0, 1, 2, 3}):
                    invalid_labels = [x for x in label_column.unique() if x not in {0, 1, 2, 3}]
                    raise ValueError(f"标签列包含无效值: {invalid_labels}。必须只包含0, 1, 2, 3")
            else:
                raise ValueError("处理后的数据框为空，请检查Excel文件格式和数据内容")

            # 分离特征和标签
            self.labels = self.dataframe.iloc[:, 0].values
            self.features = self.dataframe.iloc[:, 1:].values

            # 打印数据统计信息
            print(f"数据加载完成: {len(self.dataframe)} 样本, {self.features.shape[1]} 特征")
            print(f"类别分布: {dict(pd.Series(self.labels).value_counts())}")

        except Exception as e:
            raise RuntimeError(f"处理Excel文件时出错: {str(e)}") from e

    def _normalize_features(self):
        """标准化特征"""
        print("标准化特征...")
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def _balance_classes(self):
        """平衡类别分布"""
        print("平衡类别分布...")
        method = self.preprocessing['balance_method']

        if method == 'oversample':
            # 使用SMOTE过采样
            smote = SMOTE(random_state=self.preprocessing['random_state'])
            self.features, self.labels = smote.fit_resample(self.features, self.labels)
        elif method == 'undersample':
            # 随机欠采样
            rus = RandomUnderSampler(random_state=self.preprocessing['random_state'])
            self.features, self.labels = rus.fit_resample(self.features, self.labels)
        else:
            raise ValueError(f"不支持的类别平衡方法: {method}")

        print(f"平衡后类别分布: {dict(pd.Series(self.labels).value_counts())}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label


# 数据加载器创建函数
def create_dataloader(excel_file, sheet_name=0, batch_size=32, shuffle=True,
                      transform=None, preprocessing=None, split=True, val_size=0.2):
    """
    创建数据加载器，支持数据集分割和高级预处理

    参数:
        excel_file: Excel文件路径
        sheet_name: 工作表名称或索引
        batch_size: 批量大小
        shuffle: 是否打乱数据
        transform: 特征转换函数
        preprocessing: 预处理选项
        split: 是否分割数据集
        val_size: 验证集比例
    """
    try:
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"文件不存在: {excel_file}")

        # 创建数据集
        dataset = ExcelDataset(excel_file, sheet_name=sheet_name,
                               transform=transform, preprocessing=preprocessing)

        if len(dataset) == 0:
            raise ValueError("创建的数据集长度为0，请检查Excel文件内容")

        if split:
            # 分割数据集
            train_idx, val_idx = train_test_split(
                range(len(dataset)),
                test_size=val_size,
                stratify=dataset.labels,
                random_state=preprocessing.get('random_state', 42) if preprocessing else 42
            )

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)

            print(f"数据集分割: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader
        else:
            # 不分割，返回单个数据加载器
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return dataloader

    except Exception as e:
        raise RuntimeError(f"创建数据加载器失败: {str(e)}") from e