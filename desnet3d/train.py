import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from data import MedicalDataset
from model import densenet169_3d
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
import pandas as pd
# 设备配置
device = torch.device('cpu')

# 初始化模型
model = densenet169_3d(num_classes=3, in_channels=1).to(device)

# 定义损失函数和优化器（带类别权重）
dataset = MedicalDataset(root_dir='E:/D/data_set/hegu')
class_counts = np.bincount(dataset.labels)  # 假设数据集有labels属性
class_weights = torch.tensor(1. / (class_counts / class_counts.max()), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

indices = np.arange(len(dataset))
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.4,
    stratify=dataset.labels,
    random_state=42
)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
# 创建带平衡采样的DataLoader
train_indices = train_dataset.indices
train_labels = [dataset.labels[i] for i in train_indices]  # 正确获取训练集标签
val_labels = [dataset.labels[i] for i in val_dataset.indices]
sample_weights = class_weights[train_labels].cpu().numpy()  # 生成样本权重

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


def calculate_specificity(cm):
    """改进的多分类特异性计算（宏平均）"""
    n_classes = 3  # 固定三类
    specificities = []

    for i in range(n_classes):
        if i >= cm.shape[0]:  # 处理缺失类别
            specificities.append(1.0)
            continue

        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        denominator = tn + fp
        specificity = tn / denominator if denominator != 0 else 1.0
        specificities.append(specificity)

    return np.mean(specificities)


def calculate_metrics(true_labels, predictions, probs):
    """统一指标计算函数"""
    # 转换并验证输入格式
    true_labels = np.array(true_labels).astype(int).flatten()
    predictions = np.array(predictions).astype(int).flatten()
    probs = np.array(probs)  # 确保probs是二维概率数组

    # 处理可能的one-hot编码
    if true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=1)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    classes = [0, 1, 2]
    metrics = {}

    # 基础分类指标（添加异常捕获）
    try:
        metrics['precision'] = precision_score(true_labels, predictions,
                                               average='macro', labels=classes, zero_division=0)
        metrics['recall'] = recall_score(true_labels, predictions,
                                         average='macro', labels=classes, zero_division=0)
        metrics['f1'] = f1_score(true_labels, predictions,
                                 average='macro', labels=classes, zero_division=0)
    except Exception as e:
        print(f"基础指标计算错误: {str(e)}")
        metrics.update({'precision': 0, 'recall': 0, 'f1': 0})

    # 混淆矩阵（添加维度验证）
    try:
        cm = confusion_matrix(true_labels, predictions, labels=classes)
        metrics['specificity'] = calculate_specificity(cm)
    except Exception as e:
        print(f"混淆矩阵错误: {str(e)}")
        metrics['specificity'] = 0

    # 修正后的AUC计算逻辑
    try:
        unique_classes = np.unique(true_labels)
        if len(unique_classes) == 1:
            # 单类别时无法计算AUC，赋予默认值
            metrics['auc'] = 1.0  # 或根据需求设为0.0
        else:
            # 验证probs格式
            assert probs.ndim == 2, "probs必须是二维概率数组"
            assert probs.shape[1] == len(classes), "probs列数需与类别数一致"
            # 计算多分类AUC
            metrics['auc'] = roc_auc_score(
                true_labels, probs, multi_class='ovr', average='macro'
            )
    except Exception as e:
        print(f"AUC计算错误: {str(e)}")
        metrics['auc'] = 0.0  # 确保即使失败也有默认值

    return metrics


def train_epoch(model, loader, optimizer, criterion):
    """训练阶段"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        # 记录数据
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

        # 损失计算
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    # 指标计算
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return running_loss / len(loader.dataset), calculate_metrics(all_labels, all_preds, all_probs)


def validate(model, loader, criterion):
    """验证阶段"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # 记录数据
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_probs.append(probs.detach().cpu())

            # 损失计算
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

    # 指标计算
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return running_loss / len(loader.dataset), calculate_metrics(all_labels, all_preds, all_probs)


def train_model(model, criterion, optimizer, num_epochs=25):
    best_auc = 0.0
    history = {'train': [], 'val': []}
    metrics_df = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion)
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_metrics["recall"]:.4f}')  # 使用recall近似准确率
        print(f'[Train] AUC: {train_metrics["auc"]:.4f} | Prec: {train_metrics["precision"]:.4f} | '
              f'Rec: {train_metrics["recall"]:.4f} | F1: {train_metrics["f1"]:.4f} | '
              f'Spec: {train_metrics["specificity"]:.4f}')

        # 验证阶段
        val_loss, val_metrics = validate(model, val_loader, criterion)
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_metrics["recall"]:.4f}')
        print(f'[Val] AUC: {val_metrics["auc"]:.4f} | Prec: {val_metrics["precision"]:.4f} | '
              f'Rec: {val_metrics["recall"]:.4f} | F1: {val_metrics["f1"]:.4f} | '
              f'Spec: {val_metrics["specificity"]:.4f}')

        epoch_record = {
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train AUC': train_metrics['auc'],
            'Train Precision': train_metrics['precision'],
            'Train Recall': train_metrics['recall'],
            'Train F1': train_metrics['f1'],
            'Train Specificity': train_metrics['specificity'],
            'Val Loss': val_loss,
            'Val AUC': val_metrics['auc'],
            'Val Precision': val_metrics['precision'],
            'Val Recall': val_metrics['recall'],
            'Val F1': val_metrics['f1'],
            'Val Specificity': val_metrics['specificity']
        }
        metrics_df.append(epoch_record)

        # 保存到Excel（新增，每个epoch都保存）
        pd.DataFrame(metrics_df).to_excel('training_metrics.xlsx', index=False)

        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), 'best_model_1.pth')
            print(f'New best model saved with AUC {best_auc:.4f}')

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

    print(f'Training complete. Best validation AUC: {best_auc:.4f}')
    return model, history


if __name__ == '__main__':
    trained_model, training_history = train_model(model, criterion, optimizer, num_epochs=100)
    torch.save(trained_model, 'final_model.pth')