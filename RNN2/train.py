import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset import ExcelDataset
from model import RNNModule  # 导入LSTM模型
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase

# 设置中文字体为Arial，同时支持负号显示
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class LSTMTrainer:
    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 device='cuda', learning_rate=1e-8):
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
            if outputs.shape[1] == 3:  # 三分类情况
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
        """训练一个epoch（调整输入维度处理）"""
        self.model.train()
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # LSTM输入维度为[batch_size, seq_len, feature_dim]
            # 假设ExcelDataset输出为[batch_size, feature_dim]，需添加seq_len维度
            # 这里假设seq_len=1（单时间步输入），如需处理序列数据需调整数据集
            inputs = inputs.unsqueeze(1)  # 调整为[batch_size, 1, feature_dim]

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
        """验证/测试模型（调整输入维度处理）"""
        self.model.eval()
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"{mode.capitalize()}ing"):
                inputs = inputs.to(self.device).unsqueeze(1)  # 添加seq_len维度
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
        """完整训练流程（包含保存最终模型功能）"""
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
                best_model_path = os.path.join(save_dir, f'best_model_epoch{epoch}.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}% 至 {best_model_path}")

        # 保存最终训练模型
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"\n所有epoch训练完成！最终模型已保存至: {final_model_path}")

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

            bootstrap_metrics = {
                'accuracy': [],
                'auc': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'specificity': []
            }

            is_multiclass = all_probs.shape[1] == 3

            for i in tqdm(range(n_bootstrap), desc=f"{dataset_name}自助采样"):
                indices = np.random.choice(len(all_labels), size=len(all_labels), replace=True)
                sample_labels = all_labels[indices]
                sample_preds = all_preds[indices]
                sample_probs = all_probs[indices]

                # 计算准确率
                acc = np.mean(sample_labels == sample_preds) * 100
                bootstrap_metrics['accuracy'].append(acc)

                # 计算AUC（多分类）
                try:
                    if is_multiclass:
                        labels_one_hot = np.eye(all_probs.shape[1])[sample_labels]
                        auc = roc_auc_score(labels_one_hot, sample_probs, multi_class='ovr', average='weighted')
                    else:
                        auc = 0.5
                except:
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
                num_classes = cm.shape[0]
                specificity = []
                for i in range(num_classes):
                    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                    fp = cm[:, i].sum() - cm[i, i]
                    specificity.append(tn / (tn + fp)) if (tn + fp) > 0 else specificity.append(0)
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

    def calculate_and_plot_shap(self, save_dir, max_samples=100, max_display=8):
        """计算并绘制SHAP值（修正多分类问题）"""
        self.model.eval()
        device = self.device
        test_loader = self.test_loader

        if test_loader is None:
            print("测试集加载器不存在，无法计算SHAP值")
            return

        # 收集测试集数据
        test_data = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device).unsqueeze(1)
                labels = labels.to(device)
                test_data.append(inputs.cpu().numpy())
                test_labels.append(labels.cpu().numpy())

        test_data = np.concatenate(test_data, axis=0).squeeze(axis=1)  # 转为[样本数, 特征数]
        test_labels = np.concatenate(test_labels, axis=0)

        # 采样数据以提高计算效率
        sample_indices = np.random.choice(len(test_data), min(max_samples, len(test_data)), replace=False)
        sampled_data = test_data[sample_indices]
        sampled_labels = test_labels[sample_indices]

        # 定义模型预测函数
        def model_prediction(x):
            x = torch.from_numpy(x).float().to(device).unsqueeze(1)
            with torch.no_grad():
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs

        # 准备训练集数据作为背景
        train_data = []
        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(device).unsqueeze(1)
                train_data.append(inputs.cpu().numpy())
        train_data = np.concatenate(train_data, axis=0).squeeze(axis=1)
        masker = shap.maskers.Independent(train_data)

        # 计算SHAP值
        explainer = shap.Explainer(model_prediction, masker)
        shap_values = explainer(sampled_data)

        # 创建保存目录
        shap_dir = os.path.join(save_dir, "shap_plots")
        os.makedirs(shap_dir, exist_ok=True)
        feature_names = ["CAL", "Periodontal pocket", "Missing teeth", "Jawbone loss",
                         "Occlusal", "looseness", "Root bifurcation lesion", "Gingival condition"]

        # 处理多分类SHAP值
        if len(shap_values.values.shape) == 3:  # 多分类情况 [n_samples, n_features, n_classes]
            # 计算每个特征的全局重要性（跨所有类别的平均绝对值）
            shap_vals_global = np.mean(np.abs(shap_values.values), axis=2)  # [n_samples, n_features]
        else:
            shap_vals_global = np.abs(shap_values.values)

        # 转换为DataFrame - 确保形状是[n_samples, n_features]
        values = pd.DataFrame(shap_vals_global.astype(np.float32), columns=feature_names)

        # 计算每个特征的平均绝对SHAP值并排序
        values_mean = values.abs().mean().sort_values(ascending=False)
        # 特征值数据（采样的测试集特征值）
        feature_values = pd.DataFrame(sampled_data.astype(np.float32), columns=feature_names)

        # 绘制原有SHAP summary plot (全局特征重要性)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_global, sampled_data, plot_type="bar",
                          feature_names=feature_names, show=False)
        plt.title("Global Feature Importance (Bar Plot)")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, "global_feature_importance.png"))
        plt.close()

        # 为每个类别单独绘制SHAP summary plot
        if len(shap_values.values.shape) == 3:
            for class_idx in range(shap_values.values.shape[2]):
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values.values[:, :, class_idx], sampled_data,
                                  plot_type="bar", feature_names=feature_names, show=False)
                plt.title(f"Class {class_idx} Feature Importance (Bar Plot)")
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, f"class_{class_idx}_feature_importance.png"))
                plt.close()

        # ---------------------- 极坐标可视化 ----------------------
        # 选取重要性最高的前max_display个特征
        top_features = values_mean.iloc[:max_display]
        selected_values = values[top_features.index].copy()
        selected_feature_values = feature_values[top_features.index].copy()

        # 极坐标参数设置
        theta = np.linspace(0.5 * np.pi, 2.5 * np.pi, max_display, endpoint=False)

        # 医疗场景专用配色方案：蓝→紫→粉→黄渐变
        colors = [
            (173 / 255, 216 / 255, 230 / 255),  # 浅蓝
            (218 / 255, 183 / 255, 230 / 255),  # 浅紫
            (255 / 255, 192 / 255, 203 / 255),  # 浅粉
            (255 / 255, 240 / 255, 180 / 255)  # 浅黄
        ]
        cmap = LinearSegmentedColormap.from_list('medical_cmap', colors, N=256)

        # 创建极坐标图形
        fig, ax_main = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # 绘制基础扇形图（特征重要性）
        colors_array = cmap(np.linspace(0, 1, max_display + 1))
        bars1 = ax_main.bar(
            theta,
            height=0.08,  # 扇形高度
            width=np.pi / max_display,  # 扇形宽度
            bottom=0,
            color=colors_array,
            edgecolor='white',
            alpha=0.5,
            align='center',
        )

        # 内层扇形（背景）
        color2 = ['ghostwhite', 'whitesmoke']
        cmap2 = ListedColormap(color2, N=max_display)
        bars2 = ax_main.bar(
            theta,
            height=[0.08 - top_features.iloc[i] / top_features.max() * 0.08 for i in range(max_display)],
            width=np.pi / max_display,
            bottom=0,
            color=cmap2.colors,
            edgecolor='white',
            align='center',
        )

        # 优化蜂巢散点重叠
        def beeswarm_offset_optimized(y_values, width_scale=1.0, point_size=0.008):
            """优化版蜂巢图算法，减少散点重叠"""
            n = len(y_values)
            if n == 0:
                return np.array([]), np.array([])

            x_offsets = np.zeros(n, dtype=np.float32)
            y_offsets = np.zeros(n, dtype=np.float32)
            placed_points = np.empty((n, 2), dtype=np.float32)
            placed_count = 0

            # 按SHAP值排序，优先放置小值点，减少重叠
            sorted_indices = np.argsort(y_values)
            y_threshold = point_size * 1.5  # 增大垂直方向碰撞阈值
            x_threshold = point_size * 2.0  # 增大水平方向碰撞阈值

            for idx in sorted_indices:
                y_val = y_values[idx]
                center_available = True

                # 检查中心位置是否可用
                if placed_count > 0:
                    y_dists = np.abs(placed_points[:placed_count, 0] - y_val)
                    x_dists = np.abs(placed_points[:placed_count, 1])
                    collision_mask = (y_dists < y_threshold) & (x_dists < x_threshold)
                    center_available = not np.any(collision_mask)

                if center_available:
                    best_x = 0.0
                else:
                    best_x = None
                    # 动态步长搜索：从大到小尝试偏移
                    for step in np.linspace(0.01, 0.05, 5)[::-1]:
                        for side in [1, -1]:
                            test_x = side * step
                            collision = False
                            for p in range(placed_count):
                                y_dist = abs(placed_points[p, 0] - y_val)
                                x_dist = abs(placed_points[p, 1] - test_x)
                                if y_dist < y_threshold and x_dist < x_threshold:
                                    collision = True
                                    break
                            if not collision:
                                best_x = test_x
                                break
                        if best_x is not None:
                            break
                    # 若找不到位置，强制放置
                    if best_x is None:
                        best_x = 0.05 * np.sign(np.random.randn())

                x_offsets[idx] = best_x
                y_offsets[idx] = 0.0
                placed_points[placed_count] = [y_val, best_x]
                placed_count += 1

            return x_offsets, y_offsets

        # 预处理SHAP值范围和特征值归一化
        shap_mins = np.zeros(max_display, dtype=np.float32)
        shap_maxs = np.zeros(max_display, dtype=np.float32)
        feature_norms = []

        for i in range(max_display):
            shap_v = selected_values.iloc[:, i]
            feature_v = selected_feature_values.iloc[:, i]
            shap_mins[i] = shap_v.min()
            shap_maxs[i] = shap_v.max()
            # 归一化特征值到[0,1]
            feature_min, feature_max = feature_v.min(), feature_v.max()
            feature_v_norm = (feature_v - feature_min) / (feature_max - feature_min + 1e-8)
            feature_norms.append(feature_v_norm.values)

        # 绘制蜂巢图散点
        for i in range(max_display):
            shap_v = selected_values.iloc[:, i]
            feature_v_norm = feature_norms[i]
            # 优化SHAP值到半径的映射，避免边缘拥挤
            shap_min, shap_max = shap_mins[i], shap_maxs[i]
            r = (shap_v - shap_min) / (shap_max - shap_min + 1e-8) * 0.06 + 0.10  # 范围[0.10, 0.16]

            # 计算扇形宽度
            bar_width = top_features.iloc[i] / top_features.sum() * 2 * np.pi

            # 计算偏移（使用优化后的算法）
            x_offsets, y_offsets = beeswarm_offset_optimized(r.values, width_scale=3.5, point_size=0.008)

            # 添加随机抖动，进一步减少重叠
            t = theta[i] + x_offsets * bar_width / 5 + np.random.normal(0, 0.001, size=len(x_offsets))
            r_adjusted = r + y_offsets * 0.01 + np.random.normal(0, 0.001, size=len(y_offsets))

            # 绘制散点（颜色表示特征值高低）
            ax_main.scatter(
                t, r_adjusted,
                c=feature_v_norm,
                cmap=cmap,
                s=35,  # 减小点大小
                alpha=0.8,
                edgecolors='none',
                linewidth=0.1
            )

        # 添加辅助线和标签
        # 中心虚线圆
        center_radius = 0.125
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        r_circle = np.full_like(theta_circle, center_radius)
        ax_main.plot(theta_circle, r_circle, color='k', linestyle='--', linewidth=0.2)

        # 特征名称标签
        feature_label_radius = 0.18
        for i in range(max_display):
            feature_name = top_features.index[i]
            angle = theta[i]
            rotation_angle = np.degrees(angle)
            if np.pi < angle < 2 * np.pi:
                rotation_angle += 180
            ax_main.text(
                angle, feature_label_radius,
                feature_name,
                ha='center', va='center',
                fontsize=10,
                rotation=rotation_angle - 90,
                rotation_mode='anchor'
            )

        # 特征重要性数值标签
        importance_label_radius = 0.075
        for i in range(max_display):
            mean_value = values_mean.iloc[i]
            angle = theta[i]
            rotation_angle = np.degrees(angle)
            if np.pi < angle < 2 * np.pi:
                rotation_angle += 180
            ax_main.text(
                angle, importance_label_radius,
                f'{mean_value:.3f}',
                ha='center', va='center',
                fontsize=10,
                rotation=rotation_angle - 90,
                rotation_mode='anchor'
            )

        # 添加SHAP值刻度线
        tick_length = 0.02
        label_offset = 0.04
        for i in range(max_display):
            shap_min, shap_max = shap_mins[i], shap_maxs[i]
            right_angle = theta[i]

            # 计算刻度位置
            r_min = (shap_min - shap_min) / (shap_max - shap_min + 1e-8) * 0.075 + 0.09
            r_0 = (0 - shap_min) / (shap_max - shap_min + 1e-8) * 0.075 + 0.09
            r_max = (shap_max - shap_min) / (shap_max - shap_min + 1e-8) * 0.075 + 0.09

            # 主刻度线
            ax_main.plot([right_angle, right_angle], [r_min, r_max], color='k', linewidth=0.6, alpha=1)

            # 刻度标记
            tick_angles = np.array([right_angle - tick_length, right_angle + tick_length])
            ax_main.plot(tick_angles, [r_min, r_min], color='k', linewidth=0.5, alpha=1)
            ax_main.plot(tick_angles, [r_0, r_0], color='k', linewidth=0.5, alpha=1)
            ax_main.plot(tick_angles, [r_max, r_max], color='k', linewidth=0.5, alpha=1)

            # 刻度标签
            label_angle = right_angle - label_offset
            if np.pi < right_angle < 2 * np.pi:
                rotation = np.degrees(label_angle) + 90
                ha = 'right'
            else:
                rotation = np.degrees(label_angle) - 90
                ha = 'left'

            ax_main.text(
                label_angle, r_min, f'{shap_min:.2f}',
                ha=ha, va='center', fontsize=8,
                rotation=rotation, rotation_mode='anchor'
            )
            ax_main.text(
                label_angle, r_0, '0.0',
                ha=ha, va='center', fontsize=8,
                rotation=rotation, rotation_mode='anchor'
            )
            ax_main.text(
                label_angle, r_max, f'{shap_max:.2f}',
                ha=ha, va='center', fontsize=8,
                rotation=rotation, rotation_mode='anchor'
            )

        # 设置图形样式
        ax_main.set_xticklabels([])
        ax_main.set_yticklabels([])
        ax_main.grid(False)
        ax_main.spines[:].set_visible(False)
        ax_main.set_facecolor('none')

        # 添加颜色条（表示特征值高低）
        cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.015])  # 调整位置避免重叠
        norm = Normalize(vmin=0, vmax=1)
        cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
        cbar.set_label('Feature Value', labelpad=-5, fontsize=12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.ax.tick_params(labelsize=10, length=0)
        cbar.outline.set_visible(False)

        # 保存极坐标图
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, "shap_radar_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP极坐标图已保存至: {os.path.join(shap_dir, 'shap_radar_plot.png')}")

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
        """绘制三分类校准曲线"""
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

            colors = ['#B7DBE3', '#BCAFFF', '#A7AEE2']
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
            colors = ['#88C4D7', '#D0EAD5', '#AFADD2']

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
            colors = ['#88C4D7', '#D0EAD5', '#AFADD2']

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

            # 图例设置：Arial字体，大小22，无边框，向右移动0.5mm
            # 使用bbox_to_anchor调整位置，相对于axes坐标系
            plt.legend(loc="lower left",
                       bbox_to_anchor=(0.02, 0),  # 向右移动约0.5mm
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
            colors = ['#88C4D7', '#D0EAD5', '#AFADD2']
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

            # 图例设置：Arial字体，大小22，无边框
            plt.legend(loc='lower left', prop={'family': 'Arial', 'size': 22}, frameon=False)

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
            plt.savefig(os.path.join(save_dir_current, f"{dataset_type}_decision_curve.png"), dpi=300,
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


# 使用示例
if __name__ == "__main__":
    # 1. 准备三分类数据
    excel_file = 'E:/D/lin.xlsx'
    dataset = ExcelDataset(excel_file, sheet_name=0, transform=None)

    # 划分训练集、验证集、测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. 初始化LSTM模型（三分类）
    input_feature_size = train_dataset[0][0].shape[0]
    num_classes = 3  # 三分类
    hidden_size = 32
    num_layers = 2
    dropout = 0.3

    model = RNNModule(
        input_size=input_feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )

    # 3. 训练
    trainer = LSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-4
    )

    history = trainer.train(epochs=100)
