import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
from data import MedicalDataset
from model import densenet121_3d

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metrics(true_labels, predictions, probas):
    """计算各项评估指标"""
    try:
        auc = roc_auc_score(true_labels, probas, multi_class='ovr')
    except ValueError:
        auc = 0.0

    return {
        'auc': auc,
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(true_labels, predictions, average='macro', zero_division=0),
        'f1': f1_score(true_labels, predictions, average='macro', zero_division=0),
        'specificity': specificity_score(true_labels, predictions)
    }

def specificity_score(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    spec_scores = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec_scores.append(tn / (tn + fp) if (tn + fp) != 0 else 0.0)
    return np.mean(spec_scores)


def online_learn_and_evaluate(test_data_dir, pretrained_path, num_epochs=100):
    # 初始化模型
    model = densenet121_3d(num_classes=3, in_channels=1).to(device)

    # 加载预训练权重
    try:
        model.load_state_dict(torch.load(pretrained_path))
        print(f"成功加载预训练权重：{pretrained_path}")
    except Exception as e:
        print(f"权重加载失败: {str(e)}")
        print("将从头开始训练")

    # 加载测试集（保持shuffle=True）
    test_dataset = MedicalDataset(root_dir=test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    # 计算类别权重（基于测试集分布）
    class_counts = np.bincount(test_dataset.labels)
    class_weights = torch.tensor(1. / (class_counts / class_counts.max()),
                                 dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 定义优化器（使用更小学习率进行微调）
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)  # 调小学习率

    # 存储结果
    results = []
    best_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        # === 动态学习阶段 ===
        model.train()
        epoch_losses = []

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播更新参数
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # === 评估阶段 ===
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算指标
        avg_loss = np.mean(epoch_losses)
        metrics = calculate_metrics(all_labels, all_preds, all_probs)

        # 记录结果
        epoch_result = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'auc': metrics['auc'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'specificity': metrics['specificity']
        }
        results.append(epoch_result)

        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), f'E:/D/desnet3d/best_model.pth')
            print(f"发现新的最佳模型，已保存至 finetuned_best_model.pth")

        # 打印进度
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(
            f"Train Loss: {avg_loss:.4f} | Test Acc: {metrics['accuracy']:.4f} | Test F1: {metrics['f1']:.4f} | Test AUC: {metrics['auc']:.4f} | Test precision: {metrics['precision']:.4f} | Test recall: {metrics['recall']:.4f} | Test specificity: {metrics['specificity']:.4f} ")
        print("=" * 50)

    # 保存最终模型和结果
    torch.save(model.state_dict(), 'E:/D/desnet3d/best_model.pth')
    df = pd.DataFrame(results)
    df.to_excel("finetuning_results.xlsx", index=False)
    print("\n微调结果已保存至 finetuning_results.xlsx")


if __name__ == '__main__':
    TEST_DATA_DIR = 'E:/D/data_set/hegu'
    PRETRAINED_PATH = 'E:/D/desnet3d/best_model_1.pth'  # 预训练模型路径

    online_learn_and_evaluate(
        test_data_dir=TEST_DATA_DIR,
        pretrained_path=PRETRAINED_PATH,
        num_epochs=100
    )