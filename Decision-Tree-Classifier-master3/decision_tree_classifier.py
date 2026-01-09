from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os
import shap
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from scipy.stats import bootstrap
from sklearn.model_selection import StratifiedKFold

# 设置Arial字体，统一图表字体
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ------------------- 新增：混淆矩阵绘制函数（修改版）-------------------
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, set_name="", normalize=False, save_path=None, cmap=plt.cm.Blues):
    """
    绘制按标签大小排序的混淆矩阵，统一图表样式
    """
    # 按标签大小升序排序
    sorted_indices = np.argsort(classes)
    sorted_classes = np.array(classes)[sorted_indices]
    cm = cm[sorted_indices][:, sorted_indices]  # 重新排列混淆矩阵行列

    plt.figure(figsize=(10, 8))  # 调整为统一尺寸

    # 处理归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'{set_name} Normalized Confusion Matrix'
    else:
        title = f'{set_name} Confusion Matrix'

    # 绘制矩阵
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontname='Arial', fontsize=25)  # 统一字体和大小

    # 设置坐标轴标签和刻度（使用排序后的类别）
    tick_marks = np.arange(len(sorted_classes))
    plt.xticks(tick_marks, sorted_classes, rotation=45, fontname='Arial', fontsize=22)
    plt.yticks(tick_marks, sorted_classes, fontname='Arial', fontsize=22)

    # 添加文本标注
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(len(sorted_classes)):
        for j in range(len(sorted_classes)):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=20,
                     color="white" if cm[i, j] > thresh else "black")

    # 优化布局和标签
    plt.ylabel('True Label', fontname='Arial', fontsize=25)
    plt.xlabel('Predicted Label', fontname='Arial', fontsize=25)
    plt.tight_layout()

    # 仅保留左侧和下方坐标轴，设置线宽为2
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 文件路径保持不变
file_path = 'E:/D/lin.xlsx'

# 文件存在性检查
if os.path.exists(file_path):
    print("File exists")
    datasets = pd.read_excel(file_path, header=0)
else:
    print("File does not exist, please check the path and file name")
    exit()

# 数据预处理
Y = datasets.iloc[:, 0].astype(int)  # 确保标签为整数类型（0-3）
X = datasets.iloc[:, 1:]
feature_names = X.columns.tolist()
n_classes = len(Y.unique())  # 自动获取类别数（此处应为4）
class_names = Y.unique().tolist()  # 新增：获取类别名称（如0,1,2,3）

# 划分数据集：训练集70%，验证集15%，测试集15%
X_temp, X_Test, Y_temp, Y_Test = train_test_split(X, Y, test_size=0.15, random_state=0)
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_temp, Y_temp, test_size=(0.15 / 0.85),
                                                  random_state=0)  # 精确计算：0.15/(1-0.15)

# 特征标准化
sc_X = StandardScaler()
X_Train = pd.DataFrame(sc_X.fit_transform(X_Train), columns=feature_names)
X_Val = pd.DataFrame(sc_X.transform(X_Val), columns=feature_names)
X_Test = pd.DataFrame(sc_X.transform(X_Test), columns=feature_names)


def calculate_metrics(y_true, y_pred, y_proba, set_name="", n_classes=4, class_names=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    # ------------------- 调用混淆矩阵绘制函数 -------------------
    plot_dir = f"{set_name}_confusion_matrix"
    os.makedirs(plot_dir, exist_ok=True)
    plot_confusion_matrix(cm,
                          classes=class_names,
                          set_name=set_name,
                          save_path=f"{plot_dir}/confusion_matrix_{set_name}.png")  # 保存非归一化矩阵
    plot_confusion_matrix(cm,
                          classes=class_names,
                          set_name=f"{set_name}_normalized",
                          normalize=True,
                          save_path=f"{plot_dir}/confusion_matrix_{set_name}_normalized.png")  # 保存归一化矩阵

    # 初始化存储每个类别指标的列表
    precisions = []
    recalls = []
    f1s = []
    specificities = []

    # 计算每个类别的指标并存储（使用排序后的标签）
    sorted_indices = np.argsort(class_names)
    sorted_classes = np.array(class_names)[sorted_indices]

    for class_idx in sorted_indices:
        tp = cm[class_idx, class_idx]
        fn = np.sum(cm[class_idx, :]) - tp
        fp = np.sum(cm[:, class_idx]) - tp
        tn = np.sum(cm) - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        specificities.append(specificity)

    # 计算宏平均
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    macro_specificity = np.mean(specificities)

    metrics = {
        f"{set_name}_Accuracy": accuracy_score(y_true, y_pred),
        f"{set_name}_Macro_Precision": macro_precision,
        f"{set_name}_Macro_Recall": macro_recall,
        f"{set_name}_Macro_F1": macro_f1,
        f"{set_name}_Macro_Specificity": macro_specificity
    }

    try:
        metrics[f"{set_name}_AUC"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", labels=range(n_classes)
        )
    except ValueError as e:
        if "Only one class present" in str(e):
            print(f"Warning: {set_name} set has only one class, AUC set to NaN")
            metrics[f"{set_name}_AUC"] = np.nan
        else:
            raise e

    # 打印结果（使用排序后的类别）
    print(f"\n{set_name} Metrics:")
    print(f"  Accuracy: {metrics[f'{set_name}_Accuracy']:.4f}")
    if np.isnan(metrics[f'{set_name}_AUC']):
        print("  AUC: Invalid (single class data)")
    else:
        print(f"  AUC (OVR): {metrics[f'{set_name}_AUC']:.4f}")
    print(f"  Macro Precision: {metrics[f'{set_name}_Macro_Precision']:.4f}")
    print(f"  Macro Recall: {metrics[f'{set_name}_Macro_Recall']:.4f}")
    print(f"  Macro F1: {metrics[f'{set_name}_Macro_F1']:.4f}")
    print(f"  Macro Specificity: {metrics[f'{set_name}_Macro_Specificity']:.4f}")

    return metrics


# 实验不同树数量的影响
n_estimators_list = range(1, 101, 1)

# 创建DataFrame保存所有指标（调整为宏平均）
columns = [
    'n_estimators',
    'Train_Accuracy', 'Train_AUC', 'Train_Macro_Precision', 'Train_Macro_Recall',
    'Train_Macro_F1', 'Train_Macro_Specificity',
    'Val_Accuracy', 'Val_AUC', 'Val_Macro_Precision', 'Val_Macro_Recall',
    'Val_Macro_F1', 'Val_Macro_Specificity',
    'Test_Accuracy', 'Test_AUC', 'Test_Macro_Precision', 'Test_Macro_Recall',
    'Test_Macro_F1', 'Test_Macro_Specificity'
]
results_df = pd.DataFrame(columns=columns)

for n_est in n_estimators_list:
    print(f"\nTraining with n_estimators = {n_est}...")

    rf_classifier = RandomForestClassifier(
        n_estimators=n_est,
        criterion='entropy',
        random_state=0,
        max_depth=10,
        min_samples_leaf=10,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf_classifier.fit(X_Train, Y_Train)

    # 预测
    Y_Train_Pred = rf_classifier.predict(X_Train)
    y_train_proba = rf_classifier.predict_proba(X_Train)
    Y_Val_Pred = rf_classifier.predict(X_Val)
    y_val_proba = rf_classifier.predict_proba(X_Val)
    Y_Test_Pred = rf_classifier.predict(X_Test)
    y_test_proba = rf_classifier.predict_proba(X_Test)

    train_metrics = calculate_metrics(Y_Train, Y_Train_Pred, y_train_proba,
                                      "Train", n_classes, class_names=class_names)
    val_metrics = calculate_metrics(Y_Val, Y_Val_Pred, y_val_proba,
                                    "Val", n_classes, class_names=class_names)
    test_metrics = calculate_metrics(Y_Test, Y_Test_Pred, y_test_proba,
                                     "Test", n_classes, class_names=class_names)

    # 合并指标到DataFrame
    row = {
        'n_estimators': n_est,
        'Train_Accuracy': train_metrics['Train_Accuracy'],
        'Train_AUC': train_metrics['Train_AUC'],
        'Train_Macro_Precision': train_metrics['Train_Macro_Precision'],
        'Train_Macro_Recall': train_metrics['Train_Macro_Recall'],
        'Train_Macro_F1': train_metrics['Train_Macro_F1'],
        'Train_Macro_Specificity': train_metrics['Train_Macro_Specificity'],
        'Val_Accuracy': val_metrics['Val_Accuracy'],
        'Val_AUC': val_metrics['Val_AUC'],
        'Val_Macro_Precision': val_metrics['Val_Macro_Precision'],
        'Val_Macro_Recall': val_metrics['Val_Macro_Recall'],
        'Val_Macro_F1': val_metrics['Val_Macro_F1'],
        'Val_Macro_Specificity': val_metrics['Val_Macro_Specificity'],
        'Test_Accuracy': test_metrics['Test_Accuracy'],
        'Test_AUC': test_metrics['Test_AUC'],
        'Test_Macro_Precision': test_metrics['Test_Macro_Precision'],
        'Test_Macro_Recall': test_metrics['Test_Macro_Recall'],
        'Test_Macro_F1': test_metrics['Test_Macro_F1'],
        'Test_Macro_Specificity': test_metrics['Test_Macro_Specificity']
    }
    results_df.loc[len(results_df)] = row

    print(f"Current progress ({n_est}/100)")
    print("-" * 50)

# 保存结果到Excel（列名已调整）
output_file = "random_forest_metrics_macro.xlsx"
if os.path.exists(output_file):
    os.remove(output_file)

wb = Workbook()
ws = wb.active
for r in dataframe_to_rows(results_df, index=False, header=True):
    ws.append(r)
wb.save(output_file)
print(f"\nMetrics saved to {output_file}")

# 可视化指标变化（以Accuracy为例）
plt.figure(figsize=(10, 8))  # 调整为统一尺寸
plt.plot(results_df['n_estimators'], results_df['Train_Accuracy'], '--', label='Train Accuracy', lw=5)
plt.plot(results_df['n_estimators'], results_df['Val_Accuracy'], ':', label='Validation Accuracy', lw=5)
plt.plot(results_df['n_estimators'], results_df['Test_Accuracy'], '-', label='Test Accuracy', lw=5)
plt.title('Accuracy vs Number of Trees', fontname='Arial', fontsize=25)
plt.xlabel('Number of Trees', fontname='Arial', fontsize=25)
plt.ylabel('Accuracy Score', fontname='Arial', fontsize=25)
plt.xticks(n_estimators_list[::10], fontname='Arial', fontsize=22)
plt.yticks(fontname='Arial', fontsize=22)
plt.grid(True, alpha=0.2)

# 仅保留左侧和下方坐标轴，设置线宽为2
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# 图例设置
plt.legend(loc="best", prop={'family': 'Arial', 'size': 22}, frameon=False)
plt.tight_layout()
plt.savefig('accuracy_plot_macro.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存最终模型（选择最优n_estimators后，此处假设最优为最后一个训练的模型）
joblib.dump(rf_classifier, 'random_forest_model_final_macro.pkl', compress=True)
# ================= 修改后：校准曲线绘制 =================
print("\n=== Plotting Calibration Curves ===")

# 创建输出目录
calibration_dir = "calibration_curves"
os.makedirs(calibration_dir, exist_ok=True)

# 获取各数据集预测概率
y_train_proba = rf_classifier.predict_proba(X_Train)
y_val_proba = rf_classifier.predict_proba(X_Val)
y_test_proba = rf_classifier.predict_proba(X_Test)

# 初始化LabelBinarizer并拟合测试集标签
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(Y_Test)
n_classes = y_test_binarized.shape[1]

# 二值化其他数据集标签
y_train_binarized = lb.transform(Y_Train)
y_val_binarized = lb.transform(Y_Val)

# 定义数据集信息
datasets = [
    ('Training Set', y_train_proba, y_train_binarized),
    ('Validation Set', y_val_proba, y_val_binarized),
    ('Test Set', y_test_proba, y_test_binarized)
]

# 存储所有结果
all_results = []
max_size = 100  # 最大点大小

# 计算所有分箱数据
for data_name, y_proba, y_binarized in datasets:
    for class_idx in range(n_classes):
        y_true_class = y_binarized[:, class_idx]
        y_proba_class = y_proba[:, class_idx]

        # 计算校准曲线
        prob_true, prob_pred = calibration_curve(
            y_true_class,
            y_proba_class,
            n_bins=10,
            strategy='quantile'
        )

        # 获取分箱统计 (使用与calibration_curve相同的分箱策略)
        bin_edges = np.percentile(y_proba_class, np.linspace(0, 100, len(prob_pred) + 1))
        bin_counts = np.histogram(y_proba_class, bins=bin_edges)[0]

        # 确保点大小数组长度与数据点一致
        if len(bin_counts) > len(prob_pred):
            bin_counts = bin_counts[:len(prob_pred)]  # 截断以匹配数据点数量
        elif len(bin_counts) < len(prob_pred):
            # 扩展数组以匹配数据点数量 (通常不会发生)
            bin_counts = np.pad(bin_counts, (0, len(prob_pred) - len(bin_counts)), 'constant')

        all_results.append({
            'data_name': data_name,
            'class_idx': class_idx,
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'bin_counts': bin_counts
        })

# 计算全局最大样本量
max_bin_count_all = max([max(r['bin_counts']) for r in all_results])

# 统一颜色方案
colors = ['#88C4D7', '#D0EAD5', '#AFADD2']

# 定义数据集样式
dataset_styles = {
    'Training Set': {'color': colors[0], 'marker': 'o', 'ls': '-'},
    'Validation Set': {'color': colors[1], 'marker': 's', 'ls': '--'},
    'Test Set': {'color': colors[2], 'marker': '^', 'ls': ':'}
}

# 绘制每个类别的子图并分别保存
for class_idx in range(n_classes):
    fig, ax = plt.subplots(figsize=(10, 8))  # 统一尺寸
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', lw=5)  # 增加线宽

    # 绘制每个数据集
    for data_name in ['Training Set', 'Validation Set', 'Test Set']:
        # 筛选当前数据集和类别的结果
        results = [r for r in all_results if
                   (r['data_name'] == data_name) &
                   (r['class_idx'] == class_idx)]
        if not results:
            continue
        result = results[0]
        style = dataset_styles[data_name]

        # 计算点大小，确保与数据点数量一致
        point_sizes = (result['bin_counts'] / max_bin_count_all * max_size).astype(int)

        # 绘制散点
        ax.scatter(
            result['prob_pred'],
            result['prob_true'],
            s=point_sizes,
            edgecolor='w',
            alpha=1,
            color=style['color'],
            marker=style['marker'],
            label=data_name
        )

        # 绘制连接线，增加线宽
        ax.plot(
            result['prob_pred'],
            result['prob_true'],
            color=style['color'],
            linestyle=style['ls'],
            lw=5,
            alpha=0.6
        )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Predicted Probability (Mean)', fontname='Arial', fontsize=25)
    ax.set_ylabel('True Proportion', fontname='Arial', fontsize=25)
    ax.set_title(f'Class {class_idx} Calibration Curve', fontname='Arial', fontsize=25)

    # 设置刻度字体
    ax.tick_params(axis='both', which='major', labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')

    # 图例设置
    ax.legend(loc='lower right', prop={'family': 'Arial', 'size': 22}, frameon=False)

    # 只保留左轴和下轴，设置线宽
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(
        f"{calibration_dir}/class_{class_idx}_calibration_curve_train_val_test.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

# 输出分箱诊断信息
for data_name, _, _ in datasets:
    for class_idx in range(n_classes):
        results = [r for r in all_results if
                   (r['data_name'] == data_name) &
                   (r['class_idx'] == class_idx)]
        if not results:
            continue
        result = results[0]

        print(f"\n=== {data_name} - Class {class_idx} Bin Diagnosis ===")
        for i in range(len(result['prob_pred'])):
            print(f"Bin {i + 1}: Sample Count={result['bin_counts'][i]:<5} | "
                  f"Predicted Mean={result['prob_pred'][i]:.2f} | "
                  f"True Proportion={result['prob_true'][i]:.2f}")
print("Enhanced calibration curves saved to calibration_curves/ directory")

# ================= 导出特征数据用于箱型图 =================
boxplot_data_dir = "boxplot_data"
os.makedirs(boxplot_data_dir, exist_ok=True)
output_boxplot_file = os.path.join(boxplot_data_dir, "feature_boxplot_data.xlsx")

# 合并所有数据集的特征和标签（训练集+验证集+测试集）
X_all = pd.concat([X_Train, X_Val, X_Test], ignore_index=True)
Y_all = pd.concat([Y_Train, Y_Val, Y_Test], ignore_index=True)

# 添加标签列到特征数据中
X_all_with_label = X_all.copy()
X_all_with_label['label'] = Y_all.astype(str)  # 转换为字符串便于Excel分组

# 重塑数据为长格式（melt），便于箱型图绘制
melted_data = X_all_with_label.melt(id_vars='label', var_name='feature', value_name='value')

# 写入Excel（每个特征一个工作表，按标签分组）
with pd.ExcelWriter(output_boxplot_file, engine='openpyxl') as writer:
    # 按特征分组导出
    for feature in feature_names:
        feature_data = melted_data[melted_data['feature'] == feature]
        feature_data.to_excel(writer, sheet_name=feature, index=False)

    # 导出完整长格式数据（可选）
    melted_data.to_excel(writer, sheet_name='All_Data', index=False)

print(f"\n箱型图所需数据已保存至：{output_boxplot_file}")
print("数据格式说明：")
print(" - 每个特征对应一个工作表，包含'label'（类别）和'value'（特征值）两列")
print(" - 'All_Data'工作表包含所有特征的长格式数据，可直接用于箱型图绘制")
# ================= SHAP分析（全局特征重要性版本）=================
shap_dir = "shap_analysis_results_global"
os.makedirs(shap_dir, exist_ok=True)
final_model = rf_classifier
X_explain = X_Test
feature_names = X.columns.tolist()

print("\n=== SHAP Analysis Started ===")
print(f"Number of features in model: {final_model.n_features_in_}")
print(f"Shape of X_explain: {X_explain.shape}")
print(f"Length of feature name list: {len(feature_names)}")

# 验证特征名称一致性
if hasattr(final_model, 'feature_names_in_'):
    model_features = final_model.feature_names_in_.tolist()
    print(f"Feature names stored in model: {model_features}")

    if list(model_features) != feature_names:
        print("Warning: Feature names do not match, using model stored feature names")
        feature_names = model_features

# 创建SHAP解释器
try:
    print("\nCreating SHAP explainer...")
    explainer = shap.TreeExplainer(final_model)
    print("SHAP explainer created successfully")
except Exception as e:
    print(f"Failed to create SHAP explainer: {e}")
    print("Trying approximate method...")
    explainer = shap.TreeExplainer(final_model, data=shap.sample(X_explain, 100))
    print("SHAP explainer created successfully using alternative method")

# 计算SHAP值
print("\nCalculating SHAP values...")
try:
    shap_values = explainer.shap_values(X_explain)
    print(f"SHAP values calculated successfully, type: {type(shap_values)}")
except Exception as e:
    print(f"Failed to calculate SHAP values: {e}")
    print("Trying with background data...")
    background = shap.sample(X_Train, 100)
    shap_values = explainer.shap_values(X_explain, background)
    print("SHAP values calculated successfully using background data")

# 处理多类别SHAP值，计算全局特征重要性
print("\nCalculating global feature importance...")
if isinstance(shap_values, list):
    print(f"Multi-class SHAP values ({len(shap_values)} classes)")
    # 对每个类别计算平均绝对SHAP值后取平均
    mean_abs_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
    global_shap_values = np.mean(mean_abs_per_class, axis=0)
else:
    # 处理二分类或回归
    print("Binary classification/regression SHAP values")
    global_shap_values = np.abs(shap_values).mean(axis=0)

print(f"Global SHAP values shape: {global_shap_values.shape}")

# 生成可视化图表
try:
    print("\nGenerating visualizations...")

    # 全局特征重要性水平条形图
    plt.figure(figsize=(10, 8))
    sorted_idx = global_shap_values.argsort()[::-1]
    plt.barh(np.arange(len(sorted_idx)), global_shap_values[sorted_idx], height=0.8, color=colors[0])
    plt.yticks(np.arange(len(sorted_idx)), [feature_names[i] for i in sorted_idx], fontname='Arial', fontsize=22)
    plt.xlabel("Mean Absolute SHAP Value", fontname='Arial', fontsize=25)
    plt.title("Global Feature Importance Rank", fontname='Arial', fontsize=25)
    plt.xticks(fontname='Arial', fontsize=22)

    # 仅保留左侧和下方坐标轴，设置线宽
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{shap_dir}/global_feature_importance_rank.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Global feature importance rank plot saved")

    # SHAP特征效应散点图（自动处理多类别）
    plt.figure(figsize=(14, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        class_names=explainer.expected_value if hasattr(explainer, 'class_names') else None
    )
    plt.title("Feature Effects Distribution", fontname='Arial', fontsize=25)

    # 设置字体和坐标轴
    plt.gca().set_xlabel(plt.gca().get_xlabel(), fontname='Arial', fontsize=25)
    plt.gca().set_ylabel(plt.gca().get_ylabel(), fontname='Arial', fontsize=25)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(22)

    # 仅保留左侧和下方坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(f"{shap_dir}/feature_effects_distribution.png", dpi=300)
    plt.close()
    print("Feature effects distribution plot saved")

    # 组合热力图（使用所有类别的平均SHAP值）
    plt.figure(figsize=(14, 10))
    top_n = min(20, len(feature_names))
    top_indices = global_shap_values.argsort()[-top_n:][::-1]

    # 获取合并后的SHAP值（多类别时取平均）
    if isinstance(shap_values, list):
        combined_shap = np.mean(shap_values, axis=0)
    else:
        combined_shap = shap_values

    shap.summary_plot(
        combined_shap,
        X_explain.iloc[:, top_indices],
        feature_names=np.array(feature_names)[top_indices],
        plot_type="heatmap",
        show=False
    )
    plt.title("Top 20 Feature Heatmap", fontname='Arial', fontsize=25)

    # 设置字体
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(18)

    plt.tight_layout()
    plt.savefig(f"{shap_dir}/top20_feature_heatmap.png", dpi=300)
    plt.close()
    print("Top20 feature heatmap saved")

except Exception as e:
    print(f"Chart generation failed: {e}")
print(f"\nSHAP analysis completed, results saved to {shap_dir}/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_decision_curve(y_true, y_proba, classes, set_name="", save_path=None, colors=None,
                        n_bootstraps=2000, confidence_level=0.95):
    """绘制决策曲线（保留左/下坐标轴，曲线宽度=5，包含交点计算）"""
    plt.figure(figsize=(10, 8))
    n_classes = len(classes)

    # 创建列表存储交点结果（包含横纵坐标）
    intersection_data = []

    # 转换为numpy数组避免索引问题
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    n_samples = len(y_true)
    max_threshold = 1.0
    thresholds = np.linspace(0, max_threshold, 4000)
    bootstrapped_benefits = np.zeros((n_bootstraps, n_classes, len(thresholds)))

    # Bootstrap抽样循环
    np.random.seed(42)
    for boot_idx in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, size=n_samples, dtype=int)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]

        for class_idx in range(n_classes):
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

    # 计算统计量和绘制图表
    mean_benefit = bootstrapped_benefits.mean(axis=0)
    ci_lower = np.percentile(bootstrapped_benefits, (1 - confidence_level) / 2 * 100, axis=0)
    ci_upper = np.percentile(bootstrapped_benefits, (1 + confidence_level) / 2 * 100, axis=0)

    for class_idx in range(n_classes):
        color = colors[class_idx % len(colors)] if colors else f"C{class_idx}"
        class_mean_benefit = mean_benefit[class_idx]

        # 绘制平均净收益曲线（加粗至5）
        plt.plot(thresholds, class_mean_benefit, lw=5, color=color,
                 label=f'Class {classes[class_idx]}')
        # 绘制置信区间（浅填充）
        plt.fill_between(thresholds, ci_lower[class_idx], ci_upper[class_idx],
                         color=color, alpha=0.15)

        # 计算与Treat None的交点（y=0）
        none_sign_diff = np.diff(np.sign(class_mean_benefit))
        none_intersect_indices = np.where(none_sign_diff != 0)[0]

        # 处理无交点的情况：设置为0
        if len(none_intersect_indices) == 0:
            intersection_data.append({
                'Dataset': set_name,
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
                    'Dataset': set_name,
                    'Class': f'Class {class_idx}',
                    'Intersection_Type': 'Treat None',
                    'Threshold_X': round(x_intersect, 4),
                    'Net_Benefit_Y': round(y_intersect, 4)
                })

        # 计算与Treat All的交点
        prevalence = np.mean(y_true == class_idx)
        treat_all_benefit = prevalence - (1 - prevalence) * thresholds
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
                'Dataset': set_name,
                'Class': f'Class {class_idx}',
                'Intersection_Type': 'Treat All',
                'Threshold_X': round(x_intersect, 4),
                'Net_Benefit_Y': round(y_intersect, 4)
            })

    # 计算总体患病率
    prevalence = np.mean([(y_true == i).mean() for i in range(n_classes)])
    treat_all_benefit = prevalence - (1 - prevalence) * thresholds

    # 绘制参考线（宽度设为5）
    plt.axhline(y=0, color='black', linestyle='--', label='Treat none', lw=5)
    plt.plot(thresholds, treat_all_benefit, color='navy', linestyle=':',
             label='Treat all', alpha=0.7, lw=5)

    plt.xlim([0, max_threshold])
    plt.ylim([-0.05, np.max(mean_benefit) + 0.05])

    # 字体配置（Arial+大小25）
    plt.xlabel('Threshold Probability', fontname='Arial', fontsize=25)
    plt.ylabel('Net Benefit', fontname='Arial', fontsize=25)
    plt.title(f'Decision Curve ({set_name})', fontname='Arial', fontsize=25)

    # 坐标轴刻度字体配置
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(22)

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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # 保存交点结果到Excel
        if intersection_data:
            # 提取保存路径的目录
            save_dir = os.path.dirname(save_path)
            # 创建Excel文件路径
            excel_path = os.path.join(save_dir, f"{set_name.lower()}_decision_curve_intersections.xlsx")
            intersection_df = pd.DataFrame(intersection_data)
            intersection_df = intersection_df.sort_values(by=['Dataset', 'Class', 'Intersection_Type'])
            intersection_df.to_excel(excel_path, index=False)
            print(f"决策曲线交点结果已保存至: {excel_path}")

    plt.close()


# ================= 修改后的ROC曲线绘制函数 =================
def plot_roc_curves(y_true_binarized, y_proba, classes, set_name="", save_path=None, colors=None):
    """绘制多分类ROC曲线（保留左/下坐标轴，曲线宽度=5）"""
    plt.figure(figsize=(10, 8))

    # 确保classes按数值排序，保证Class 0,1,2的顺序
    sorted_classes = sorted(classes)
    n_classes = len(sorted_classes)

    # 按排序后的类别绘制ROC曲线（保证图例顺序）
    for sorted_idx, class_label in enumerate(sorted_classes):
        # 找到原始索引
        class_idx = list(classes).index(class_label)
        fpr, tpr, _ = roc_curve(y_true_binarized[:, class_idx], y_proba[:, class_idx])
        roc_auc = auc(fpr, tpr)
        color = colors[sorted_idx % len(colors)] if colors else f"C{sorted_idx}"
        plt.plot(fpr, tpr, color=color, lw=5,
                 label=f'Class {class_label} (AUC = {roc_auc:.4f})')

    # 对角线参考线：宽度设为5
    plt.plot([0, 1], [0, 1], 'k--', lw=5)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    # 字体配置（Arial+大小25）
    plt.xlabel('False Positive Rate', fontname='Arial', fontsize=25)
    plt.ylabel('True Positive Rate', fontname='Arial', fontsize=25)
    plt.title(f'ROC Curve ({set_name})', fontname='Arial', fontsize=25)

    # 坐标轴刻度字体配置
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(22)

    # 图例设置：Arial字体，大小22，无边框，向左移动
    plt.legend(loc="lower right",
               bbox_to_anchor=(0.98, 0.02),
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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ================= 修改后的PR曲线绘制函数 =================
def plot_pr_curves(y_true_binarized, y_proba, classes, set_name="", save_path=None, colors=None):
    """绘制多分类PR曲线（保留左/下坐标轴，曲线宽度=5）"""
    plt.figure(figsize=(10, 8))

    # 确保classes按数值排序，保证Class 0,1,2的顺序
    sorted_classes = sorted(classes)
    n_classes = len(sorted_classes)

    # 按排序后的类别绘制PR曲线（保证图例顺序）
    for sorted_idx, class_label in enumerate(sorted_classes):
        # 找到原始索引
        class_idx = list(classes).index(class_label)
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, class_idx], y_proba[:, class_idx])
        ap = np.trapz(precision, recall)  # 计算AP值
        color = colors[sorted_idx % len(colors)] if colors else f"C{sorted_idx}"
        plt.plot(recall, precision, color=color, lw=5,
                 label=f'Class {class_label} (AP = {ap:.4f})')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    # 字体配置（Arial+大小25）
    plt.xlabel('Recall', fontname='Arial', fontsize=25)
    plt.ylabel('Precision', fontname='Arial', fontsize=25)
    plt.title(f'Precision-Recall Curve ({set_name})', fontname='Arial', fontsize=25)

    # 坐标轴刻度字体配置
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(22)

    # 图例设置：Arial字体，大小22，无边框
    plt.legend(loc="lower left",
               bbox_to_anchor=(0.02, 0),
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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ================= 在主流程中添加调用 =================
import matplotlib.pyplot as plt
import os

# 创建输出目录
dca_dir = "decision_curves"
roc_dir = "roc_curves"
pr_dir = "pr_curves"
os.makedirs(roc_dir, exist_ok=True)
os.makedirs(pr_dir, exist_ok=True)
os.makedirs(dca_dir, exist_ok=True)

# 统一颜色方案
colors = ['#88C4D7', '#D0EAD5', '#AFADD2']

# 获取各数据集的二值化标签和预测概率
datasets_info = [
    ("Train", Y_Train, y_train_proba),
    ("Val", Y_Val, y_val_proba),
    ("Test", Y_Test, y_test_proba)
]

# 初始化LabelBinarizer（确保与校准曲线部分一致）
lb = LabelBinarizer()
lb.fit(Y)  # 拟合所有标签

for set_name, y_true, y_proba in datasets_info:
    y_true_binarized = lb.transform(y_true)

    # 绘制ROC曲线
    plot_roc_curves(
        y_true_binarized,
        y_proba,
        classes=class_names,
        set_name=set_name,
        save_path=f"{roc_dir}/{set_name.lower()}_roc_curves.png",
        colors=colors  # 传递颜色参数
    )

    # 绘制PR曲线
    plot_pr_curves(
        y_true_binarized,
        y_proba,
        classes=class_names,
        set_name=set_name,
        colors=colors,
        save_path=f"{pr_dir}/{set_name.lower()}_pr_curves.png"
    )
    print(f"{set_name} ROC and PR curves saved")

    # 区分处理测试集
    if set_name == "Test":
        plot_decision_curve(y_true, y_proba, classes=class_names, set_name=set_name,
                            colors=colors,
                            save_path=f"{dca_dir}/{set_name.lower()}_decision_curve_bootstrap.png",
                            n_bootstraps=2000, confidence_level=0.95)
        print(f"{set_name} Bootstrap Decision Curve saved")
    else:
        plot_decision_curve(y_true, y_proba, classes=class_names, set_name=set_name,
                            colors=colors, save_path=f"{dca_dir}/{set_name.lower()}_decision_curve.png")
        print(f"{set_name} Decision Curve saved")


# 新增：计算指标及其置信区间的函数
def calculate_metrics_with_ci(model, X, y, dataset_name, metrics_functions, n_bootstraps=1000, ci_level=0.95):
    """
    计算指标点估计值和置信区间
    """
    results = []
    n_samples = len(y)
    rng = np.random.default_rng(42)  # 固定随机种子确保可重复性

    # 定义自助抽样函数
    def bootstrap_sample():
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    # 计算原始数据点估计
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    point_estimates = {name: func(y, y_pred, y_proba) for name, func in metrics_functions.items()}

    # 自助抽样计算置信区间
    for metric_name, metric_func in metrics_functions.items():
        bootstrap_values = []
        for _ in range(n_bootstraps):
            X_boot, y_boot = bootstrap_sample()
            y_proba_boot = model.predict_proba(X_boot)
            y_pred_boot = model.predict(X_boot)
            bootstrap_values.append(metric_func(y_boot, y_pred_boot, y_proba_boot))

        # 计算置信区间
        sorted_values = np.sort(bootstrap_values)
        lower = sorted_values[int((1 - ci_level) / 2 * n_bootstraps)]
        upper = sorted_values[int((1 + ci_level) / 2 * n_bootstraps)]

        results.append({
            'Metric': metric_name,
            'Point Estimate': point_estimates[metric_name],
            'CI Lower': lower,
            'CI Upper': upper,
            'Dataset': dataset_name
        })

    return pd.DataFrame(results)


# 定义需要计算的指标及其计算函数
metrics_functions = {
    'Accuracy': lambda y, y_pred, y_proba: accuracy_score(y, y_pred),
    'AUC': lambda y, y_pred, y_proba: roc_auc_score(y, y_proba, multi_class='ovr'),
    'Precision': lambda y, y_pred, y_proba: precision_score(y, y_pred, average='macro'),
    'Recall': lambda y, y_pred, y_proba: recall_score(y, y_pred, average='macro'),
    'F1': lambda y, y_pred, y_proba: f1_score(y, y_pred, average='macro'),
    'Specificity': lambda y, y_pred, y_proba:
    np.mean([tn / (tn + fp) if (tn + fp) != 0 else 0
             for tn, fp in
             zip(confusion_matrix(y, y_pred)[:, 0],
                 confusion_matrix(y, y_pred)[:, 1])])
}

# 计算各数据集的置信区间
# 直接传入数据集名称，而不是依赖DataFrame的name属性
ci_train = calculate_metrics_with_ci(final_model, X_Train, Y_Train, "Train", metrics_functions)
ci_val = calculate_metrics_with_ci(final_model, X_Val, Y_Val, "Validation", metrics_functions)
ci_test = calculate_metrics_with_ci(final_model, X_Test, Y_Test, "Test", metrics_functions)

# 合并结果
all_ci_results = pd.concat([ci_train, ci_val, ci_test], ignore_index=True)

# 写入单独的Excel文件
ci_output_path = 'metrics_confidence_intervals.xlsx'
with pd.ExcelWriter(ci_output_path, engine='openpyxl') as writer:
    # 创建透视表，按指标和数据集组织结果
    pivot_table = all_ci_results.pivot_table(
        index='Metric',
        columns='Dataset',
        values=['Point Estimate', 'CI Lower', 'CI Upper']
    )

    # 确保列按有意义的顺序排列
    ordered_datasets = ['Train', 'Validation', 'Test']
    ordered_columns = [(metric_type, dataset)
                       for metric_type in ['Point Estimate', 'CI Lower', 'CI Upper']
                       for dataset in ordered_datasets]

    # 重新排序列
    pivot_table = pivot_table[ordered_columns]

    # 写入Excel
    pivot_table.to_excel(writer, sheet_name='CI Results', float_format='%.4f')

    # 添加单独的工作表，按数据集分开
    for dataset in ['Train', 'Validation', 'Test']:
        dataset_results = all_ci_results[all_ci_results['Dataset'] == dataset]
        dataset_results = dataset_results[['Metric', 'Point Estimate', 'CI Lower', 'CI Upper']]
        dataset_results.to_excel(writer, sheet_name=f'CI_{dataset}', index=False, float_format='%.4f')

print(f"\n置信区间计算完成，结果已保存至 {ci_output_path}")

# 可选：打印结果摘要
print("\n=== 置信区间摘要 ===")
for dataset in ['Train', 'Validation', 'Test']:
    subset = all_ci_results[all_ci_results['Dataset'] == dataset]
    print(f"\n{dataset} 数据集:")
    for _, row in subset.iterrows():
        print(f"{row['Metric']}: {row['Point Estimate']:.4f} ({row['CI Lower']:.4f}-{row['CI Upper']:.4f})")


# ==================== 五折交叉验证函数 ====================
# 执行五折交叉验证的函数也需要相应修改
def calculate_metrics_with_cv(model, X, y, metrics_functions, n_splits=5, random_state=42):
    """
    执行五折交叉验证并计算指标
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_train_scaled, y_train)

        y_proba = model.predict_proba(X_val_scaled)
        y_pred = model.predict(X_val_scaled)

        fold_metrics = {
            "Fold": fold,
            **{name: func(y_val, y_pred, y_proba) for name, func in metrics_functions.items()}
        }
        fold_results.append(fold_metrics)

    fold_df = pd.DataFrame(fold_results).set_index("Fold")

    # 计算点估计和置信区间
    ci_results = []
    for metric in metrics_functions.keys():
        values = fold_df[metric].values
        point_estimate = np.mean(values)

        boot_result = bootstrap(
            (values,),
            statistic=lambda x: np.mean(x),
            n_resamples=1000,
            random_state=random_state
        )
        ci_lower, ci_upper = boot_result.confidence_interval

        ci_results.append({
            "Metric": metric,
            "Point Estimate": point_estimate,
            "CI Lower": ci_lower,
            "CI Upper": ci_upper,
            "Dataset": "5-Fold CV"
        })

    return pd.DataFrame(ci_results), fold_df


# ==================== 执行五折交叉验证 ====================
# 初始化模型（使用最终确定的超参数，如n_estimators=100）
cv_model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    random_state=0,
    max_depth=10,
    min_samples_leaf=10,
    n_jobs=-1,
    class_weight='balanced'
)

# 执行交叉验证并获取结果
cv_ci, cv_folds = calculate_metrics_with_cv(
    model=cv_model,
    X=X_Train,
    y=Y_Train,
    metrics_functions=metrics_functions,
    n_splits=5,
    random_state=0
)

# ==================== 保存交叉验证结果 ====================
# 创建输出文件
cv_output_file = "cv_metrics_with_ci.xlsx"
with pd.ExcelWriter(cv_output_file, engine='openpyxl') as writer:
    # 写入置信区间
    cv_ci.to_excel(writer, sheet_name="CI Summary", float_format="%.4f")

    # 写入各折叠详细结果
    cv_folds.to_excel(writer, sheet_name="Fold Results", float_format="%.4f")

print(f"\n五折交叉验证完成，结果保存至 {cv_output_file}")
print("\n各折叠指标详情:")
print(cv_folds)
print("\n置信区间摘要:")
print(cv_ci[["Metric", "Point Estimate", "CI Lower", "CI Upper"]])
