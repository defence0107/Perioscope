from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = 'E:/D/linjun.xlsx'

# 文件存在性检查
if os.path.exists(file_path):
    print("文件存在")
    datasets = pd.read_excel(file_path, header=0)
else:
    print("文件不存在，请检查路径和文件名")
    exit()

# 数据预处理
Y = datasets.iloc[:, 0]
X = datasets.iloc[:, 1:]
feature_names = X.columns.tolist()

# 检查类别分布
print("\n类别分布情况:")
print(Y.value_counts())

# 划分数据集：训练集70%，验证集15%，测试集15%
X_temp, X_Test, Y_temp, Y_Test = train_test_split(X, Y, test_size=0.15, random_state=0, stratify=Y)
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_temp, Y_temp, test_size=(0.15 / 0.85), random_state=0,
                                                  stratify=Y_temp)

# 特征标准化（逻辑回归对特征缩放比较敏感，这一步很重要）
sc_X = StandardScaler()
X_Train = pd.DataFrame(sc_X.fit_transform(X_Train), columns=feature_names)
X_Val = pd.DataFrame(sc_X.transform(X_Val), columns=feature_names)
X_Test = pd.DataFrame(sc_X.transform(X_Test), columns=feature_names)

# 实验不同正则化强度的影响
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # C是正则化强度的倒数，值越小正则化越强
best_C = None
best_val_accuracy = 0
best_model = None

# 早停机制参数
patience = 3  # 连续3次没有提升则停止
no_improve_count = 0

for C in C_values:
    print(f"\n正在训练 C = {C}...")

    # 初始化逻辑回归分类器
    lr_classifier = LogisticRegression(
        C=C,  # 正则化强度的倒数
        penalty='l2',  # L2正则化，默认值
        solver='lbfgs',  # 适合中小数据集的求解器
        max_iter=1000,  # 增加迭代次数以确保收敛
        random_state=0,
        n_jobs=-1,
        class_weight='balanced'  # 处理不平衡数据
    )

    # 训练模型
    lr_classifier.fit(X_Train, Y_Train)

    # 在验证集上评估
    Y_Val_Pred = lr_classifier.predict(X_Val)
    val_accuracy = np.mean(Y_Val_Pred == Y_Val)

    # 输出更详细的评估指标
    print("\n验证集分类报告:")
    print(classification_report(Y_Val, Y_Val_Pred))

    # 记录最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_C = C
        best_model = lr_classifier
        no_improve_count = 0  # 重置计数器
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"\n早停机制触发: 连续{patience}次未提升准确率")
            break

    print(f"当前进度, 验证集准确率: {val_accuracy:.4f}")
    print("-" * 50)

print(f"\n最佳正则化参数 C: {best_C}, 最佳验证集准确率: {best_val_accuracy:.4f}")

# 分析特征系数（逻辑回归中相当于特征重要性）
if best_model is not None:
    coefficients = best_model.coef_[0]
    indices = np.argsort(np.abs(coefficients))[::-1]  # 按绝对值排序

    # 打印特征系数
    print("\n特征系数（绝对值排序）:")
    for f in range(min(10, X.shape[1])):  # 打印前10个影响最大的特征
        coef = coefficients[indices[f]]
        print(f"{feature_names[indices[f]]}: {coef:.4f} (影响方向: {'正' if coef > 0 else '负'})")

    # 可视化特征系数
    plt.figure(figsize=(10, 6))
    plt.title('特征系数（绝对值）')
    plt.bar(range(X.shape[1]), np.abs(coefficients[indices]), align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_coefficients.png')
    print("\n特征系数图已保存为 feature_coefficients.png")

# 保存最佳模型
if best_model is not None:
    joblib.dump(best_model, 'logistic_regression_model_final.pkl', compress=False)
    joblib.dump(sc_X, 'standard_scaler.pkl', compress=False)

    print("\n最佳模型和标准化器已保存:")
    print("- logistic_regression_model_final.pkl")
    print("- standard_scaler.pkl")

    # 在测试集上评估最终模型
    print("\n测试集评估结果:")
    Y_Test_Pred = best_model.predict(X_Test)
    print(f"测试集准确率: {np.mean(Y_Test_Pred == Y_Test):.4f}")
    print("\n测试集分类报告:")
    print(classification_report(Y_Test, Y_Test_Pred))
    print("\n混淆矩阵:")
    print(confusion_matrix(Y_Test, Y_Test_Pred))

    # 验证模型加载
    try:
        loaded_model = joblib.load('logistic_regression_model_final.pkl')
        print("\n模型验证成功:")
        print(f"模型类型: {type(loaded_model)}")
        print(f"模型参数: C={loaded_model.C}")
        test_pred = loaded_model.predict(X_Test.head(1))
        print(f"测试预测结果: {test_pred}")
    except Exception as e:
        print(f"模型验证失败: {str(e)}")
else:
    print("错误: 没有找到最佳模型")
