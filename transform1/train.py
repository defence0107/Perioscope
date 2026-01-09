import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from Dataset import ExcelDataset
from model import Transformer  # 假设Transformer已支持三分类输出
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib.colors import ListedColormap
import seaborn as sns

# 设置中文字体为Arial，同时支持负号显示
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class TransformerClassifierTrainer:
    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 损失函数和优化器（三分类使用CrossEntropyLoss）
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 训练历史记录（三分类评估指标）
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_auc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'train_specificity': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_specificity': []
        }

    def _calculate_metrics(self, outputs, labels):
        """计算三分类评估指标"""
        _, predicted = torch.max(outputs.data, 1)
        probs = torch.softmax(outputs, dim=1)

        # 转换为numpy格式
        labels_np = labels.cpu().numpy()
        predicted_np = predicted.cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        # 计算准确率
        acc = (predicted == labels).sum().item() / labels.size(0)

        # 计算AUC（三分类使用ovr方式）
        try:
            if outputs.shape[1] == 4:  # 三分类情况
                # 将标签转为one-hot编码
                labels_one_hot = np.eye(outputs.shape[1])[labels_np]
                auc = roc_auc_score(labels_one_hot, probs_np, multi_class='ovr', average='weighted')
            else:
                auc = 0.5
        except Exception as e:
            print(f"AUC计算失败: {e}")
            auc = 0.5

        # 计算Precision, Recall, F1（加权平均）
        precision = precision_score(labels_np, predicted_np, average='weighted', zero_division=0)
        recall = recall_score(labels_np, predicted_np, average='weighted', zero_division=0)
        f1 = f1_score(labels_np, predicted_np, average='weighted')

        # 计算特异性
        cm = confusion_matrix(labels_np, predicted_np)
        num_classes = cm.shape[0]
        specificity = []

        for i in range(num_classes):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            if (tn + fp) > 0:
                specificity.append(tn / (tn + fp))
            else:
                specificity.append(0)

        avg_specificity = np.mean(specificity)

        return acc * 100, auc, precision, recall, f1, avg_specificity

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 添加序列维度
            inputs = inputs.unsqueeze(1)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计信息
            running_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)

        # 计算指标
        epoch_loss = running_loss / len(self.train_loader)
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        acc, auc, precision, recall, f1, specificity = self._calculate_metrics(all_outputs, all_labels)

        # 记录历史
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(acc)
        self.history['train_auc'].append(auc)
        self.history['train_precision'].append(precision)
        self.history['train_recall'].append(recall)
        self.history['train_f1'].append(f1)
        self.history['train_specificity'].append(specificity)

        return epoch_loss, acc, auc, precision, recall, f1, specificity

    def validate(self, loader, mode='val'):
        """验证/测试模型"""
        self.model.eval()
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"{mode.capitalize()}ing"):
                inputs = inputs.to(self.device).unsqueeze(1)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        # 计算指标
        epoch_loss = running_loss / len(loader)
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        acc, auc, precision, recall, f1, specificity = self._calculate_metrics(all_outputs, all_labels)

        # 记录历史
        if mode == 'val':
            self.history['val_loss'].append(epoch_loss)
            self.history['val_acc'].append(acc)
            self.history['val_auc'].append(auc)
            self.history['val_precision'].append(precision)
            self.history['val_recall'].append(recall)
            self.history['val_f1'].append(f1)
            self.history['val_specificity'].append(specificity)

        return epoch_loss, acc, auc, precision, recall, f1, specificity

    def train(self, epochs, save_dir='./saved_models'):
        """完整训练流程"""
        start_time = time.time()
        best_val_acc = 0.0

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        train_log_path = Path(save_dir) / "training_log.xlsx"
        test_log_path = Path(save_dir) / "test_results.xlsx"

        # 定义日志列结构
        train_log_columns = [
            'Epoch',
            'Train Loss', 'Train Acc', 'Train AUC', 'Train Precision',
            'Train Recall', 'Train F1', 'Train Specificity',
            'Val Loss', 'Val Acc', 'Val AUC', 'Val Precision',
            'Val Recall', 'Val F1', 'Val Specificity'
        ]

        test_log_columns = [
            'Epoch',
            'Test Loss', 'Test Acc', 'Test AUC',
            'Test Precision', 'Test Recall',
            'Test F1', 'Test Specificity'
        ]

        # 初始化日志
        train_log_df = pd.DataFrame(columns=train_log_columns)
        test_log_df = pd.DataFrame(columns=test_log_columns)

        print(f"开始训练，共 {epochs} 个epoch...")
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # 训练 + 验证
            train_metrics = self.train_epoch()
            val_metrics = self.validate(self.val_loader, 'val') if self.val_loader else (0,) * 7
            test_metrics = self.validate(self.test_loader, 'test') if self.test_loader else (0,) * 7

            # 保存训练日志
            train_new_row = {
                'Epoch': epoch,
                'Train Loss': train_metrics[0],
                'Train Acc': train_metrics[1],
                'Train AUC': train_metrics[2],
                'Train Precision': train_metrics[3],
                'Train Recall': train_metrics[4],
                'Train F1': train_metrics[5],
                'Train Specificity': train_metrics[6],
                'Val Loss': val_metrics[0] if self.val_loader else None,
                'Val Acc': val_metrics[1] if self.val_loader else None,
                'Val AUC': val_metrics[2] if self.val_loader else None,
                'Val Precision': val_metrics[3] if self.val_loader else None,
                'Val Recall': val_metrics[4] if self.val_loader else None,
                'Val F1': val_metrics[5] if self.val_loader else None,
                'Val Specificity': val_metrics[6] if self.val_loader else None
            }
            train_log_df = pd.concat([train_log_df, pd.DataFrame([train_new_row])], ignore_index=True)
            train_log_df.to_excel(train_log_path, index=False)

            # 保存测试日志
            test_new_row = {
                'Epoch': epoch,
                'Test Loss': test_metrics[0],
                'Test Acc': test_metrics[1],
                'Test AUC': test_metrics[2],
                'Test Precision': test_metrics[3],
                'Test Recall': test_metrics[4],
                'Test F1': test_metrics[5],
                'Test Specificity': test_metrics[6]
            }
            test_log_df = pd.concat([test_log_df, pd.DataFrame([test_new_row])], ignore_index=True)
            test_log_df.to_excel(test_log_path, index=False)

            # 打印信息
            print(f"\n[Epoch {epoch}]")
            print(f"训练损失: {train_metrics[0]:.4f} | 测试准确率: {test_metrics[1]:.2f}%")
            print(f"测试F1: {test_metrics[5]:.4f} | 测试AUC: {test_metrics[2]:.4f}")

            # 保存最佳模型
            if self.val_loader and val_metrics[1] > best_val_acc:
                best_val_acc = val_metrics[1]
                torch.save(self.model.state_dict(),
                           os.path.join(save_dir, f'best_model_epoch{epoch}.pth'))
                print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")

        print(f"\n总训练时间: {time.time() - start_time:.1f}秒")

        # 计算并保存置信区间
        self.calculate_confidence_intervals(save_dir)

        # 绘制评估曲线
        print("\n开始绘制评估曲线...")
        self.plot_confusion_matrix(save_dir, 'train')
        if self.val_loader:
            self.plot_confusion_matrix(save_dir, 'val')
        if self.test_loader:
            self.plot_confusion_matrix(save_dir, 'test')

        self.plot_calibration_curve(save_dir, dataset_types=['train'])
        if self.val_loader:
            self.plot_calibration_curve(save_dir, dataset_types=['val'])
        if self.test_loader:
            self.plot_calibration_curve(save_dir, dataset_types=['test'])

        self.plot_roc_curve(save_dir, dataset_types=['train'])
        if self.val_loader:
            self.plot_roc_curve(save_dir, dataset_types=['val'])
        if self.test_loader:
            self.plot_roc_curve(save_dir, dataset_types=['test'])

        self.plot_pr_curve(save_dir, dataset_types=['train'])
        if self.val_loader:
            self.plot_pr_curve(save_dir, dataset_types=['val'])
        if self.test_loader:
            self.plot_pr_curve(save_dir, dataset_types=['test'])

        self.plot_decision_curve(save_dir, dataset_types=['train'])
        if self.val_loader:
            self.plot_decision_curve(save_dir, dataset_types=['val'])
        if self.test_loader:
            self.plot_decision_curve(save_dir, dataset_types=['test'])

        self.calculate_and_plot_shap(save_dir)

        return self.history

    def calculate_confidence_intervals(self, save_dir, n_bootstrap=1000, confidence_level=0.95):
        """计算评估指标置信区间"""
        print("\n开始计算评估指标的置信区间...")
        datasets = {
            'train': (self.train_loader, '训练集'),
            'val': (self.val_loader, '验证集'),
            'test': (self.test_loader, '测试集')
        }

        ci_results = {}
        for dataset_type, (loader, dataset_name) in datasets.items():
            if loader is None:
                print(f"跳过{dataset_name}，没有可用的数据加载器")
                continue

            print(f"\n计算{dataset_name}的置信区间...")
            all_preds = []
            all_labels = []
            all_probs = []

            self.model.eval()
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"{dataset_name}预测"):
                    inputs = inputs.to(self.device).unsqueeze(1)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            # 确保是4分类问题
            num_classes = all_probs.shape[1]
            if num_classes != 4:
                print(f"警告: 检测到{num_classes}个类别，期望是4分类问题")

            bootstrap_metrics = {
                'accuracy': [],
                'auc': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'specificity': []
            }

            # 为每个类别分别计算特异性
            for class_idx in range(num_classes):
                bootstrap_metrics[f'specificity_class_{class_idx}'] = []

            for i in tqdm(range(n_bootstrap), desc=f"{dataset_name}自助采样"):
                indices = np.random.choice(len(all_labels), size=len(all_labels), replace=True)
                sample_labels = all_labels[indices]
                sample_preds = all_preds[indices]
                sample_probs = all_probs[indices]

                # 计算准确率
                acc = np.mean(sample_labels == sample_preds) * 100
                bootstrap_metrics['accuracy'].append(acc)

                # 计算AUC（4分类）
                try:
                    if num_classes == 4:
                        labels_one_hot = np.eye(num_classes)[sample_labels]
                        # 使用ovr策略和加权平均计算多分类AUC
                        auc = roc_auc_score(labels_one_hot, sample_probs, multi_class='ovr', average='weighted')
                    else:
                        # 非4分类的处理逻辑
                        labels_one_hot = np.eye(num_classes)[sample_labels]
                        auc = roc_auc_score(labels_one_hot, sample_probs, multi_class='ovr', average='weighted')
                except Exception as e:
                    print(f"计算AUC时出错: {e}")
                    auc = np.nan
                bootstrap_metrics['auc'].append(auc)

                # 计算Precision, Recall, F1
                precision = precision_score(sample_labels, sample_preds, average='weighted', zero_division=0)
                recall = recall_score(sample_labels, sample_preds, average='weighted', zero_division=0)
                f1 = f1_score(sample_labels, sample_preds, average='weighted')
                bootstrap_metrics['precision'].append(precision)
                bootstrap_metrics['recall'].append(recall)
                bootstrap_metrics['f1'].append(f1)

                # 计算特异性
                cm = confusion_matrix(sample_labels, sample_preds)
                specificity = []

                # 确保混淆矩阵是4x4的
                if cm.shape[0] < num_classes:
                    # 如果某些类别在样本中没有出现，可能会导致混淆矩阵维度不足
                    new_cm = np.zeros((num_classes, num_classes), dtype=int)
                    for i in range(min(cm.shape[0], num_classes)):
                        for j in range(min(cm.shape[1], num_classes)):
                            new_cm[i, j] = cm[i, j]
                    cm = new_cm

                for i in range(num_classes):
                    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                    fp = cm[:, i].sum() - cm[i, i]
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    specificity.append(spec)
                    bootstrap_metrics[f'specificity_class_{i}'].append(spec)

                avg_specificity = np.mean(specificity)
                bootstrap_metrics['specificity'].append(avg_specificity)

            # 计算置信区间
            alpha = 1 - confidence_level
            lower_p = alpha / 2 * 100
            upper_p = (1 - alpha / 2) * 100

            ci_results[dataset_type] = {}
            print(f"\n{dataset_name}的置信区间 ({confidence_level * 100}%):")
            for metric, values in bootstrap_metrics.items():
                values = np.array(values)[~np.isnan(np.array(values))]
                if len(values) == 0:
                    ci_results[dataset_type][metric] = {'mean': np.nan, 'lower': np.nan, 'upper': np.nan}
                    print(f"{metric}: 无法计算（没有有效数据）")
                    continue

                mean_value = np.mean(values)
                lower_bound = np.percentile(values, lower_p)
                upper_bound = np.percentile(values, upper_p)

                ci_results[dataset_type][metric] = {
                    'mean': mean_value,
                    'lower': lower_bound,
                    'upper': upper_bound
                }
                print(f"{metric}: {mean_value:.4f} [{lower_bound:.4f}, {upper_bound:.4f}]")

        # 保存置信区间结果
        self.save_confidence_intervals_to_excel(ci_results, save_dir, confidence_level)

    def save_confidence_intervals_to_excel(self, ci_results, save_dir, confidence_level):
        """保存置信区间到Excel"""
        print("\n保存置信区间结果到Excel...")
        results_df = pd.DataFrame(columns=[
            'Dataset', 'Metric', 'Mean', 'Lower Bound', 'Upper Bound', 'Confidence Interval'
        ])

        for dataset_type, metrics in ci_results.items():
            dataset_name = {'train': '训练集', 'val': '验证集', 'test': '测试集'}.get(dataset_type, dataset_type)
            for metric, values in metrics.items():
                if np.isnan(values['mean']):
                    continue
                ci_str = f"{values['mean']:.4f} [{values['lower']:.4f}, {values['upper']:.4f}]"
                results_df = pd.concat([results_df, pd.DataFrame({
                    'Dataset': [dataset_name],
                    'Metric': [metric],
                    'Mean': [values['mean']],
                    'Lower Bound': [values['lower']],
                    'Upper Bound': [values['upper']],
                    'Confidence Interval': [ci_str]
                })], ignore_index=True)

        ci_path = os.path.join(save_dir, f"confidence_intervals_{confidence_level * 100:.0f}%.xlsx")
        results_df.to_excel(ci_path, index=False)
        print(f"置信区间结果已保存至: {ci_path}")

    def calculate_and_plot_shap(self, save_dir, max_samples=100):
        """计算并绘制SHAP值"""
        self.model.eval()
        device = self.device
        test_loader = self.test_loader

        test_data = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device).unsqueeze(1)
                labels = labels.to(device)
                test_data.append(inputs.cpu().numpy())
                test_labels.append(labels.cpu().numpy())

        test_data = np.concatenate(test_data, axis=0).squeeze(axis=1)
        test_labels = np.concatenate(test_labels, axis=0)

        sample_indices = np.random.choice(len(test_data), min(max_samples, len(test_data)), replace=False)
        sampled_data = test_data[sample_indices]
        sampled_labels = test_labels[sample_indices]

        def model_prediction(x):
            x = torch.from_numpy(x).float().to(device).unsqueeze(1)
            with torch.no_grad():
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs

        train_data = []
        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(device).unsqueeze(1)
                train_data.append(inputs.cpu().numpy())
        train_data = np.concatenate(train_data, axis=0).squeeze(axis=1)
        masker = shap.maskers.Independent(train_data)

        explainer = shap.Explainer(model_prediction, masker)
        shap_values = explainer(sampled_data)

        os.makedirs(os.path.join(save_dir, "shap_plots"), exist_ok=True)
        feature_names = ["gender", "age", "weight", "BMI", "Educational level", "Annual family income", "Drink",
                         "Drinking frequency", "Smoke", "Smoking frequency", "Degree of smoking", " healthy diet",
                         " Regular diet", "trouble sleeping", "sleep disorder", "Diabetes", "Cardiovascular disease",
                         "Respiratory system diseases", "Rheumatoid arthritis", "Alzheimer's disease"]
        shap_values.feature_names = feature_names

        # 绘制SHAP图（保持原有逻辑，多分类自动处理）
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, sampled_data, plot_type="bar", feature_names=feature_names)
        plt.title("Global Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_plots/global_feature_importance.png"))
        plt.close()

        # 其他SHAP图省略，逻辑与二分类类似

    def plot_confusion_matrix(self, save_dir, dataset_type='train'):
        """绘制混淆矩阵（三分类）"""
        self.model.eval()
        device = self.device
        save_dir = os.path.join(save_dir, "confusion_matrix_plots")
        os.makedirs(save_dir, exist_ok=True)

        if dataset_type == 'train':
            loader = self.train_loader
            title_prefix = "Training"
        elif dataset_type == 'val':
            loader = self.val_loader
            title_prefix = "Validation"
        elif dataset_type == 'test':
            loader = self.test_loader
            title_prefix = "Test"
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")

        if loader is None:
            print(f"No {dataset_type} loader available.")
            return

        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"{title_prefix} Confusion Matrix"):
                inputs = inputs.to(device).unsqueeze(1)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                all_outputs.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_outputs)
        num_classes = cm.shape[0]

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{title_prefix} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes), rotation=45)
        plt.yticks(tick_marks, range(num_classes))

        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, f"{cm[i, j]}",
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_type}_confusion_matrix.png"))
        plt.close()

    def plot_calibration_curve(self, save_dir, dataset_types=['test']):
        """绘制三分类校准曲线（支持多数据集类型）"""
        self.model.eval()
        device = self.device

        for dataset_type in dataset_types:
            save_dir_current = os.path.join(save_dir, "calibration_plots")
            os.makedirs(save_dir_current, exist_ok=True)

            # 根据数据集类型选择加载器
            if dataset_type == 'train':
                loader = self.train_loader
                title_prefix = "Training"
            elif dataset_type == 'val':
                loader = self.val_loader
                title_prefix = "Validation"
            elif dataset_type == 'test':
                loader = self.test_loader
                title_prefix = "Test"
            else:
                raise ValueError("dataset_type must be 'train', 'val', or 'test'")

            if loader is None:
                print(f"No {dataset_type} loader available.")
                continue

            all_probs = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"{title_prefix} Calibration Curve"):
                    inputs = inputs.to(device).unsqueeze(1)
                    labels = labels.to(device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            num_classes = all_probs.shape[1]

            plt.figure(figsize=(10, 8))
            ax1 = plt.subplot(111)
            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

            colors = ['#B7DBE3', '#BCAFFF', '#A7AEE2', '#C4C3DE']
            for i in range(num_classes):
                class_mask = all_labels == i
                if np.sum(class_mask) == 0:
                    continue

                class_probs = all_probs[class_mask, i]
                class_labels = all_labels[class_mask]

                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true=class_labels,
                    y_prob=class_probs,
                    n_bins=10,
                    pos_label=i
                )

                ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                         label=f"Class {i} ({title_prefix})",
                         color=colors[i % len(colors)], linewidth=2)

            ax1.set_ylabel("Fraction of positives")
            ax1.set_xlabel("Mean predicted probability")
            ax1.set_title(f"{title_prefix} Calibration Curve (Multiclass)")
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.2)

            for spine in ax1.spines.values():
                spine.set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir_current, f"{dataset_type}_calibration_curve.png"))
            plt.close()

    def plot_roc_curve(self, save_dir, dataset_types=['test']):
        """绘制三分类ROC曲线（保留左/下坐标轴，曲线宽度=5）"""
        self.model.eval()
        device = self.device

        for dataset_type in dataset_types:
            save_dir_current = os.path.join(save_dir, "roc_plots")
            os.makedirs(save_dir_current, exist_ok=True)

            if dataset_type == 'train':
                loader = self.train_loader
                title_prefix = "Training"
            elif dataset_type == 'val':
                loader = self.val_loader
                title_prefix = "Validation"
            elif dataset_type == 'test':
                loader = self.test_loader
                title_prefix = "Test"
            else:
                raise ValueError("dataset_type must be 'train', 'val', or 'test'")

            if loader is None:
                print(f"No {dataset_type} loader available.")
                continue

            all_probs = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"{title_prefix} ROC Curve"):
                    inputs = inputs.to(device).unsqueeze(1)
                    labels = labels.to(device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            num_classes = all_probs.shape[1]

            plt.figure(figsize=(10, 8))
            # 更新颜色方案
            colors = ['#88C4D7', '#D0EAD5', '#AFADD2', '#E6C7DF']

            # 类别ROC曲线：宽度设为5
            for i in range(num_classes):
                binary_labels = (all_labels == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
                roc_auc = roc_auc_score(binary_labels, all_probs[:, i])

                plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=5,
                         label=f'Class {i}(AUC = {roc_auc:.4f})')

            # 对角线参考线：宽度设为5
            plt.plot([0, 1], [0, 1], 'k--', lw=5)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            # 字体配置（Arial+大小25）
            plt.xlabel('False Positive Rate', fontname='Arial', fontsize=25)
            plt.ylabel('True Positive Rate', fontname='Arial', fontsize=25)
            plt.title(f'ROC Curve ({title_prefix})',
                      fontname='Arial', fontsize=25)

            # 图例设置：Arial字体，大小22，无边框，向左移动0.5mm
            # 使用bbox_to_anchor调整位置，相对于axes坐标系
            plt.legend(loc="lower right",
                       bbox_to_anchor=(0.98, 0.02),  # 向左移动约0.5mm
                       prop={'family': 'Arial', 'size': 22},
                       frameon=False)

            # 添加网格线，透明度0.2
            plt.grid(True, alpha=0.2)

            # 仅保留左侧和下方坐标轴，设置线宽为2
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)

            # 坐标轴刻度字体配置
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontsize(25)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir_current, f"{dataset_type}_roc_curve.png"))
            plt.close()

    def plot_pr_curve(self, save_dir, dataset_types=['test']):
        """绘制三分类PR曲线（保留左/下坐标轴，曲线宽度=5）"""
        self.model.eval()
        device = self.device

        for dataset_type in dataset_types:
            save_dir_current = os.path.join(save_dir, "pr_plots")
            os.makedirs(save_dir_current, exist_ok=True)

            if dataset_type == 'train':
                loader = self.train_loader
                title_prefix = "Training"
            elif dataset_type == 'val':
                loader = self.val_loader
                title_prefix = "Validation"
            elif dataset_type == 'test':
                loader = self.test_loader
                title_prefix = "Test"
            else:
                raise ValueError("dataset_type must be 'train', 'val', or 'test'")

            if loader is None:
                print(f"No {dataset_type} loader available.")
                continue

            all_probs = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"{title_prefix} PR Curve"):
                    inputs = inputs.to(device).unsqueeze(1)
                    labels = labels.to(device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            num_classes = all_probs.shape[1]

            plt.figure(figsize=(10, 8))
            # 更新颜色方案
            colors = ['#88C4D7', '#D0EAD5', '#AFADD2', '#E6C7DF']

            # 类别PR曲线：宽度设为5
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, i], pos_label=i)
                ap = np.trapz(precision, recall)

                plt.plot(recall, precision, color=colors[i % len(colors)], lw=5,
                         label=f'Class {i} (AP = {ap:.4f})')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            # 字体配置（Arial+大小25）
            plt.xlabel('Recall', fontname='Arial', fontsize=25)
            plt.ylabel('Precision', fontname='Arial', fontsize=25)
            plt.title(f'Precision-Recall Curve ({title_prefix})',
                      fontname='Arial', fontsize=25)

            # 图例设置：Arial字体，大小22，无边框，往上移动0.5mm
            # 通过增加bbox_to_anchor的y值实现上移
            plt.legend(loc="lower left",
                       bbox_to_anchor=(0.02, 0.02),  # y值从0调整为0.02，实现上移
                       prop={'family': 'Arial', 'size': 22},
                       frameon=False)

            # 添加网格线，透明度0.2
            plt.grid(True, alpha=0.2)

            # 仅保留左侧和下方坐标轴，设置线宽为2
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)

            # 坐标轴刻度字体配置
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontsize(25)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir_current, f"{dataset_type}_pr_curve.png"))
            plt.close()

    def plot_decision_curve(self, save_dir, dataset_types=['test'],
                            threshold_range=(0, 1, 0.01), n_bootstrap=2000, ci=0.95):
        """绘制三分类决策曲线（保留左/下坐标轴，曲线宽度=5，包含交点计算）"""
        self.model.eval()
        device = self.device

        # 创建列表存储交点结果（包含横纵坐标）
        intersection_data = []

        for dataset_type in dataset_types:
            save_dir_current = os.path.join(save_dir, "decision_plots")
            os.makedirs(save_dir_current, exist_ok=True)

            # 选择数据集加载器
            if dataset_type == 'train':
                loader = self.train_loader
                title_prefix = "Training"
            elif dataset_type == 'val':
                loader = self.val_loader
                title_prefix = "Validation"
            elif dataset_type == 'test':
                loader = self.test_loader
                title_prefix = "Test"
            else:
                raise ValueError("dataset_type must be 'train', 'val', or 'test'")

            if loader is None:
                print(f"No {dataset_type} loader available.")
                continue

            # 1. 收集预测概率和真实标签
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"{title_prefix} Decision Curve"):
                    inputs = inputs.to(device).unsqueeze(1)
                    labels = labels.to(device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            num_classes = all_probs.shape[1]
            classes = list(range(num_classes))
            # 统一颜色方案
            colors = ['#88C4D7', '#D0EAD5', '#AFADD2', '#E6C7DF']
            thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])
            n_samples = len(all_labels)

            # 2. Bootstrap抽样计算净收益
            bootstrapped_benefits = np.zeros((n_bootstrap, num_classes, len(thresholds)))
            np.random.seed(42)

            for boot_idx in range(n_bootstrap):
                indices = np.random.randint(0, n_samples, size=n_samples, dtype=int)
                y_true_boot = all_labels[indices]
                y_proba_boot = all_probs[indices]

                for class_idx in range(num_classes):
                    y_true_class = (y_true_boot == class_idx).astype(int)
                    y_proba_class = y_proba_boot[:, class_idx]

                    net_benefit = np.zeros_like(thresholds)
                    for t_idx, threshold in enumerate(thresholds):
                        y_pred = (y_proba_class >= threshold).astype(int)
                        tp = np.sum((y_pred == 1) & (y_true_class == 1))
                        fp = np.sum((y_pred == 1) & (y_true_class == 0))

                        denominator = 1 - threshold if threshold != 1 else 1e-8
                        net_benefit[t_idx] = (tp / n_samples) - (fp / n_samples) * (threshold / denominator)

                    bootstrapped_benefits[boot_idx, class_idx, :] = net_benefit

            # 3. 计算平均净收益和置信区间
            mean_benefit = bootstrapped_benefits.mean(axis=0)
            ci_lower = np.percentile(bootstrapped_benefits, (1 - ci) / 2 * 100, axis=0)
            ci_upper = np.percentile(bootstrapped_benefits, (1 + ci) / 2 * 100, axis=0)

            # 4. 计算参考线（Treat None/Treat All）
            prevalence = np.mean([(all_labels == i).mean() for i in range(num_classes)])
            treat_all_benefit = prevalence - (1 - prevalence) * thresholds
            treat_none_benefit = np.zeros_like(thresholds)

            # 5. 绘制决策曲线（统一图表风格）
            plt.figure(figsize=(10, 8))
            for class_idx in range(num_classes):
                color = colors[class_idx % len(colors)]
                class_mean_benefit = mean_benefit[class_idx]

                # 绘制平均净收益曲线（加粗至5，统一颜色）
                plt.plot(thresholds, class_mean_benefit, lw=5, color=color,
                         label=f'Class {classes[class_idx]}')
                # 绘制置信区间（浅填充、统一颜色）
                plt.fill_between(thresholds, ci_lower[class_idx], ci_upper[class_idx],
                                 color=color, alpha=0.15)

                # 计算与Treat None的交点（y=0）
                none_sign_diff = np.diff(np.sign(class_mean_benefit - treat_none_benefit))
                none_intersect_indices = np.where(none_sign_diff != 0)[0]

                # 处理无交点的情况：设置为0
                if len(none_intersect_indices) == 0:
                    intersection_data.append({
                        'Dataset': dataset_type,
                        'Class': f'Class {class_idx}',
                        'Intersection_Type': 'Treat None',
                        'Threshold_X': 0.0,  # 无交点时设置为0
                        'Net_Benefit_Y': 0.0
                    })
                else:
                    for idx in none_intersect_indices:
                        if idx >= len(thresholds) - 1:
                            continue

                        x1, x2 = thresholds[idx], thresholds[idx + 1]
                        y1, y2 = class_mean_benefit[idx], class_mean_benefit[idx + 1]
                        # 线性插值计算交点
                        x_intersect = x1 - (y1 * (x2 - x1)) / (y2 - y1) if (y2 - y1) != 0 else x1
                        y_intersect = 0.0

                        # 记录交点数据
                        intersection_data.append({
                            'Dataset': dataset_type,
                            'Class': f'Class {class_idx}',
                            'Intersection_Type': 'Treat None',
                            'Threshold_X': round(x_intersect, 4),
                            'Net_Benefit_Y': round(y_intersect, 4)
                        })

                # 计算与Treat All的交点
                all_sign_diff = np.diff(np.sign(class_mean_benefit - treat_all_benefit))
                all_intersect_indices = np.where(all_sign_diff != 0)[0]

                for idx in all_intersect_indices:
                    if idx >= len(thresholds) - 1:
                        continue

                    x1, x2 = thresholds[idx], thresholds[idx + 1]
                    y1_model, y2_model = class_mean_benefit[idx], class_mean_benefit[idx + 1]
                    y1_all, y2_all = treat_all_benefit[idx], treat_all_benefit[idx + 1]

                    # 解方程求交点
                    numerator = (y1_all - y1_model) * (x2 - x1)
                    denominator = (y2_model - y1_model) - (y2_all - y1_all)
                    if denominator == 0:
                        continue

                    x_intersect = x1 + numerator / denominator
                    # 计算交点纵坐标
                    y_intersect = y1_all + (x_intersect - x1) * (y2_all - y1_all) / (x2 - x1)

                    # 记录交点数据
                    intersection_data.append({
                        'Dataset': dataset_type,
                        'Class': f'Class {class_idx}',
                        'Intersection_Type': 'Treat All',
                        'Threshold_X': round(x_intersect, 4),
                        'Net_Benefit_Y': round(y_intersect, 4)
                    })

            # 绘制参考线（宽度设为5，统一风格）
            plt.axhline(y=0, color='black', linestyle='--', label='Treat none', lw=5)
            plt.plot(thresholds, treat_all_benefit, color='navy', linestyle=':',
                     label='Treat all', alpha=0.7, lw=5)

            plt.xlim([threshold_range[0], threshold_range[1]])
            plt.ylim([-0.05, np.max(mean_benefit) + 0.05])

            # 字体配置（Arial+大小25，统一风格）
            plt.xlabel('Threshold Probability', fontname='Arial', fontsize=25)
            plt.ylabel('Net Benefit', fontname='Arial', fontsize=25)
            plt.title(f'Decision Curve ({title_prefix}) ',
                      fontname='Arial', fontsize=25)

            # 图例设置：Arial字体，大小22，无边框，往上移动0.5mm
            plt.legend(loc='lower left',
                       bbox_to_anchor=(0.01, 0.03),  # y值从0调整为0.02，实现上移
                       prop={'family': 'Arial', 'size': 22},
                       frameon=False)

            # 添加网格线，透明度0.3
            plt.grid(True, alpha=0.3, linestyle='--')

            # 仅保留左侧和下方坐标轴，设置线宽为2
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)

            # 坐标轴刻度字体配置
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontsize(25)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir_current, f"{dataset_type}_decision_curve.png"), dpi=600,
                        bbox_inches='tight')
            plt.close()

        # 6. 保存交点结果到Excel（含横纵坐标）
        if intersection_data:
            intersection_df = pd.DataFrame(intersection_data)
            # 按数据集、类别、交点类型排序
            intersection_df = intersection_df.sort_values(by=['Dataset', 'Class', 'Intersection_Type'])
            excel_path = os.path.join(save_dir, "decision_curve_intersections_with_xy.xlsx")
            intersection_df.to_excel(excel_path, index=False)
            print(f"\n决策曲线交点结果（含横纵坐标）已保存至: {excel_path}")
        else:
            print("\n未检测到决策曲线与参考线的交点")

    def _calculate_decision_curve_with_ci(self, probs, labels, class_idx, thresholds, n_bootstrap, ci):
        """计算决策曲线并通过自助法获取置信区间"""
        n_samples = len(labels)
        net_benefit_bootstrap = np.zeros((n_bootstrap, len(thresholds)))

        # 自助法重采样计算净获益分布
        for boot_idx in range(n_bootstrap):
            # 有放回抽样
            indices = np.random.randint(0, n_samples, n_samples)
            boot_probs = probs[indices]
            boot_labels = labels[indices]

            for t_idx, t in enumerate(thresholds):
                # 模型预测的净获益
                predicted_positive = (boot_probs[:, class_idx] >= t).astype(int)
                tp = np.sum(predicted_positive * (boot_labels == class_idx))
                fp = np.sum(predicted_positive * (boot_labels != class_idx))
                net_benefit = tp - fp * (t / (1 - t)) if t != 1 else 0
                net_benefit_bootstrap[boot_idx, t_idx] = net_benefit

        # 计算置信区间
        alpha = (1 - ci) / 2
        net_benefit_mean = np.mean(net_benefit_bootstrap, axis=0)
        net_benefit_lower = np.percentile(net_benefit_bootstrap, alpha * 100, axis=0)
        net_benefit_upper = np.percentile(net_benefit_bootstrap, (1 - alpha) * 100, axis=0)

        return net_benefit_mean, net_benefit_lower, net_benefit_upper

    def _plot_single_class_decision_curve(self, save_dir, dataset_type, class_idx,
                                          threshold_range, n_bootstrap, ci):
        """为单个分类绘制决策曲线（可选辅助函数）"""
        self.model.eval()
        device = self.device

        save_dir_current = os.path.join(save_dir, "decision_plots")
        os.makedirs(save_dir_current, exist_ok=True)

        # 根据数据集类型选择加载器
        if dataset_type == 'train':
            loader = self.train_loader
            title_prefix = "Training"
        elif dataset_type == 'val':
            loader = self.val_loader
            title_prefix = "Validation"
        elif dataset_type == 'test':
            loader = self.test_loader
            title_prefix = "Test"
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")

        if loader is None:
            print(f"No {dataset_type} loader available.")
            return

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"{title_prefix} Decision Curve Class {class_idx}"):
                inputs = inputs.to(device).unsqueeze(1)
                labels = labels.to(device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        colors = ['#B7DBE3', '#BCAFFF', '#A7AEE2']

        # 生成阈值范围
        thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])

        # 计算决策曲线和置信区间
        net_benefit_mean, net_benefit_lower, net_benefit_upper = self._calculate_decision_curve_with_ci(
            all_probs, all_labels, class_idx, thresholds, n_bootstrap, ci
        )

        plt.figure(figsize=(8, 6))

        # 绘制均值曲线和置信区间
        plt.plot(thresholds, net_benefit_mean, color=colors[class_idx], lw=2,
                 label=f'Class {class_idx} Model')
        plt.fill_between(thresholds, net_benefit_lower, net_benefit_upper,
                         color=colors[class_idx], alpha=0.2, label=f'95% CI')

        # 绘制参考线
        all_positive = np.mean(all_labels == class_idx)
        plt.plot(thresholds, all_positive * np.ones_like(thresholds), 'k--', lw=2, label='All Positive')
        plt.plot(thresholds, np.zeros_like(thresholds), 'k-', lw=2, label='All Negative')

        plt.xlim(threshold_range[0], threshold_range[1])
        plt.ylim(bottom=0)
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title(f'{title_prefix} Decision Curve Analysis (Class {class_idx})')
        plt.legend(loc="lower right")
        plt.grid(True)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir_current,
                                 f"{dataset_type}_decision_curve_class{class_idx}.png"))
        plt.close()


# 使用示例
if __name__ == "__main__":
    # 1. 准备三分类数据
    excel_file = 'E:/D/shujubiao.xlsx'
    dataset = ExcelDataset(excel_file, sheet_name=0, transform=None)

    # 划分训练集、验证集、测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. 初始化Transformer模型（三分类）
    input_feature_size = train_dataset[0][0].shape[0]
    num_classes = 4  # 三分类
    d_model = 64
    num_heads = 4
    num_layers = 3
    dff = 128
    pe_input = 5000

    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_feature_size=input_feature_size,
        num_classes=num_classes,
        pe_input=pe_input
    )

    # 3. 训练
    trainer = TransformerClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001
    )

    history = trainer.train(epochs=100)
