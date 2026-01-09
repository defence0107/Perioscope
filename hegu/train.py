import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
from data import MedicalDataset
from model import resnet18_3d
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 创建保存结果的目录
os.makedirs('results', exist_ok=True)

# 初始化模型
model = resnet18_3d(num_classes=3, in_channels=1).to(device)


# 新增：加载预训练权重函数
def load_pretrained_weights(model, weights_path):
    """
    加载预训练权重
    :param model: 模型实例
    :param weights_path: 权重文件路径
    :return: 加载了预训练权重的模型
    """
    if os.path.exists(weights_path):
        print(f"加载预训练权重: {weights_path}")
        try:
            # 加载权重
            state_dict = torch.load(weights_path, map_location=device)

            # 处理可能的权重键名不匹配问题
            model_state_dict = model.state_dict()

            # 检查键名是否匹配
            if set(state_dict.keys()) == set(model_state_dict.keys()):
                model.load_state_dict(state_dict)
                print("权重加载成功！")
            else:
                print("检测到键名不匹配，尝试适配权重...")
                # 尝试适配权重（处理可能的键名前缀差异）
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 移除可能的模块前缀（如'module.'）
                    if k.startswith('module.'):
                        new_key = k[7:]  # 移除'module.'
                    else:
                        new_key = k

                    if new_key in model_state_dict:
                        new_state_dict[new_key] = v
                    else:
                        print(f"跳过不匹配的键: {k}")

                # 加载适配后的权重
                if new_state_dict:
                    model.load_state_dict(new_state_dict, strict=False)
                    print(f"部分权重加载成功（严格模式关闭）")
                else:
                    print("无法适配权重，将从头开始训练")
        except Exception as e:
            print(f"权重加载失败: {str(e)}")
            print("将从头开始训练")
    else:
        print(f"权重文件不存在: {weights_path}，将从头开始训练")

    return model


# 定义数据集和数据加载
def get_dataloaders(batch_size=8):
    # 加载完整数据集
    dataset = MedicalDataset(root_dir='D:/project/raw_dataset/hegus')
    # 数据划分 - 增加随机性检查
    indices = np.arange(len(dataset))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.3,  # 40%用于验证和测试
        stratify=dataset.labels,
        random_state=43
    )

    # 进一步将临时集划分为验证集和测试集
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,  # 验证集和测试集各占20%
        stratify=[dataset.labels[i] for i in temp_indices],
        random_state=44
    )

    # 创建子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 计算类别权重和采样权重
    class_counts = np.bincount(dataset.labels)
    class_weights = torch.tensor(1. / (class_counts / class_counts.max()), dtype=torch.float32).to(device)

    # 为训练集创建加权采样器
    train_labels = [dataset.labels[i] for i in train_indices]
    sample_weights = class_weights[train_labels].cpu().numpy()

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True  # 丢弃最后一个不完整的批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    return train_loader, val_loader, test_loader, class_weights, \
        train_labels, [dataset.labels[i] for i in val_indices]


# 获取数据加载器
batch_size = 8  # 增加批量大小
train_loader, val_loader, test_loader, class_weights, train_labels, val_labels = get_dataloaders(batch_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)  # 提高初始学习率

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # 跟踪AUC最大化
    factor=0.5,  # 学习率降低因子
    patience=8,  # 多少个epoch没有改善后降低学习率
    min_lr=1e-8,  # 最小学习率
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
    """统一指标计算函数，增加了准确率指标"""
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

    # 计算准确率
    try:
        metrics['accuracy'] = accuracy_score(true_labels, predictions)
    except Exception as e:
        print(f"准确率计算错误: {str(e)}")
        metrics['accuracy'] = 0

    # 基础分类指标
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

    # 混淆矩阵
    try:
        cm = confusion_matrix(true_labels, predictions, labels=classes)
        metrics['specificity'] = calculate_specificity(cm)
    except Exception as e:
        print(f"混淆矩阵错误: {str(e)}")
        metrics['specificity'] = 0

    # AUC计算
    try:
        unique_classes = np.unique(true_labels)
        if len(unique_classes) == 1:
            metrics['auc'] = 1.0  # 单类别时AUC设为1.0
        else:
            assert probs.ndim == 2, "probs必须是二维概率数组"
            assert probs.shape[1] == len(classes), "probs列数需与类别数一致"
            metrics['auc'] = roc_auc_score(
                true_labels, probs, multi_class='ovr', average='macro'
            )
    except Exception as e:
        print(f"AUC计算错误: {str(e)}")
        metrics['auc'] = 0.0

    return metrics


def train_epoch(model, loader, optimizer, criterion):
    """训练阶段，修复梯度范数的设备转换问题"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    grad_norms = []

    for batch_idx, (inputs, labels) in enumerate(loader):
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

        # 记录梯度范数 - 修复设备转换问题
        grad_norm = clip_grad_norm_(model.parameters(), 5.0)
        # 将CUDA张量转换为CPU后再添加到列表
        grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # 每10个批次打印一次进度
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}')

    # 计算平均梯度范数
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0

    # 指标计算
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return running_loss / len(loader.dataset), calculate_metrics(all_labels, all_preds, all_probs), avg_grad_norm


def validate(model, loader, criterion):
    """验证阶段，增加了详细的批次信息"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
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


def plot_training_history(history):
    """绘制训练历史曲线"""
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'specificity']

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # 绘制训练指标
        if metric == 'loss':
            plt.plot([h['loss'] for h in history['train']], label='训练')
        else:
            plt.plot([h[metric] for h in history['train']], label='训练')

        # 绘制验证指标
        if metric == 'loss':
            plt.plot([h['loss'] for h in history['val']], label='验证')
        else:
            plt.plot([h[metric] for h in history['val']], label='验证')

        plt.title(f'{metric} 变化曲线')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{metric}_curve.png')
        plt.close()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, start_epoch=0):
    best_auc = 0.0
    history = {
        'train': [],
        'val': []
    }

    # 新增：如果从中间开始训练，尝试加载之前的最佳模型
    if start_epoch > 0:
        best_model_path = 'results/best_model.pth'
        if os.path.exists(best_model_path):
            print(f"从第{start_epoch}轮开始训练，加载之前的最佳模型...")
            model = load_pretrained_weights(model, best_model_path)

    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'当前学习率: {current_lr:.8f}')

        # 训练阶段
        train_loss, train_metrics, avg_grad_norm = train_epoch(model, train_loader, optimizer, criterion)
        print(f'\n训练集损失: {train_loss:.4f} | 平均梯度范数: {avg_grad_norm:.4f}')
        print(f'[训练] 准确率: {train_metrics["accuracy"]:.4f} | AUC: {train_metrics["auc"]:.4f}')
        print(f'[训练] 精确率: {train_metrics["precision"]:.4f} | 召回率: {train_metrics["recall"]:.4f}')
        print(f'[训练] F1值: {train_metrics["f1"]:.4f} | 特异性: {train_metrics["specificity"]:.4f}')

        # 验证阶段
        val_loss, val_metrics = validate(model, val_loader, criterion)
        print(f'\n验证集损失: {val_loss:.4f}')
        print(f'[验证] 准确率: {val_metrics["accuracy"]:.4f} | AUC: {val_metrics["auc"]:.4f}')
        print(f'[验证] 精确率: {val_metrics["precision"]:.4f} | 召回率: {val_metrics["recall"]:.4f}')
        print(f'[验证] F1值: {val_metrics["f1"]:.4f} | 特异性: {val_metrics["specificity"]:.4f}')

        # 记录历史
        history['train'].append({**train_metrics, 'loss': train_loss})
        history['val'].append({**val_metrics, 'loss': val_loss})

        # 学习率调度
        scheduler.step(val_metrics['auc'])

        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), 'results/best_model.pth')
            print(f'新的最佳模型已保存，AUC: {best_auc:.4f}')

        # 新增：定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'history': history
            }
            torch.save(checkpoint, f'results/checkpoint_epoch_{epoch + 1}.pth')
            print(f'检查点已保存: results/checkpoint_epoch_{epoch + 1}.pth')

    # 绘制训练历史
    plot_training_history(history)

    # 训练结束后在测试集上评估
    print(f'\n训练完成。最佳验证集AUC: {best_auc:.4f}')
    print('在测试集上评估最佳模型...')

    # 加载最佳模型权重
    model.load_state_dict(torch.load('results/best_model.pth'))
    test_loss, test_metrics = validate(model, test_loader, criterion)

    print(f'\n测试集损失: {test_loss:.4f}')
    print(f'[测试] 准确率: {test_metrics["accuracy"]:.4f} | AUC: {test_metrics["auc"]:.4f}')
    print(f'[测试] 精确率: {test_metrics["precision"]:.4f} | 召回率: {test_metrics["recall"]:.4f}')
    print(f'[测试] F1值: {test_metrics["f1"]:.4f} | 特异性: {test_metrics["specificity"]:.4f}')

    # 保存测试集结果
    with open('results/test_metrics.txt', 'w') as f:
        for key, value in test_metrics.items():
            f.write(f'{key}: {value:.4f}\n')
        f.write(f'loss: {test_loss:.4f}\n')

    return model, history


def save_metrics_to_excel(history, save_path='results/training_metrics.xlsx'):
    """
    将训练历史中的指标写入Excel表格
    :param history: 训练历史字典，格式为 {'train': [epoch1_metrics, ...], 'val': [epoch1_metrics, ...]}
    :param save_path: Excel保存路径（默认保存在results目录下）
    """
    # 1. 定义统一的指标顺序（确保训练/验证指标列一致）
    metric_order = ['epoch', 'loss', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'specificity']

    # 2. 处理训练集指标：添加epoch列，按统一顺序整理
    train_data = history['train'].copy()
    # 为每个epoch添加序号（从1开始）
    for i in range(len(train_data)):
        train_data[i]['epoch'] = i + 1  # epoch从1开始，符合直觉
    # 转换为DataFrame并按指标顺序排序
    train_df = pd.DataFrame(train_data)[metric_order]

    # 3. 处理验证集指标：同上
    val_data = history['val'].copy()
    for i in range(len(val_data)):
        val_data[i]['epoch'] = i + 1
    val_df = pd.DataFrame(val_data)[metric_order]

    # 4. 写入Excel（分两个工作表：Train和Val）
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        train_df.to_excel(writer, sheet_name='Train', index=False)  # 不保存行索引
        val_df.to_excel(writer, sheet_name='Val', index=False)

    print(f"\n训练/验证指标已成功写入Excel：{save_path}")
    print(f"训练集指标行数：{len(train_df)}（对应{len(train_df)}个epoch）")
    print(f"验证集指标行数：{len(val_df)}（对应{len(val_df)}个epoch）")


# 新增：加载检查点继续训练的函数
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    加载检查点继续训练
    :param model: 模型实例
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param checkpoint_path: 检查点文件路径
    :return: 模型、优化器、调度器、当前epoch、最佳auc、历史记录
    """
    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        best_auc = checkpoint['best_auc']
        history = checkpoint['history']

        print(f"从第{start_epoch}轮继续训练，之前的最佳AUC: {best_auc:.4f}")
        return model, optimizer, scheduler, start_epoch, best_auc, history
    else:
        print(f"检查点文件不存在: {checkpoint_path}")
        return model, optimizer, scheduler, 0, 0.0, {'train': [], 'val': []}


if __name__ == '__main__':
    # 新增：选择训练模式
    train_mode = input("选择训练模式 (1: 从头开始, 2: 加载预训练权重, 3: 继续训练): ")

    start_epoch = 0
    best_auc = 0.0
    training_history = {'train': [], 'val': []}

    if train_mode == '2':
        # 加载预训练权重
        weights_path = input("请输入预训练权重文件路径: ").strip()
        model = load_pretrained_weights(model, weights_path)
    elif train_mode == '3':
        # 继续训练
        checkpoint_path = input("请输入检查点文件路径: ").strip()
        model, optimizer, scheduler, start_epoch, best_auc, training_history = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path
        )

    # 训练模型
    trained_model, history = train_model(
        model, criterion, optimizer, scheduler,
        num_epochs=150, start_epoch=start_epoch
    )

    torch.save(trained_model.state_dict(), 'results/final_model.pth')
    np.save('results/training_history.npy', history)

    # 将指标写入Excel2

    save_metrics_to_excel(
        history=history,
        save_path='results/training_metrics.xlsx'
    )