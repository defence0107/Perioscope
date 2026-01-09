import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from model import Transformer  # 假设Transformer模型和其他相关函数已经定义
from Dataset import ExcelDataset
import pandas as pd

# 测试模块
def test_model(model_path, test_data_path, output_path):
    # 设置超参数
    NUM_LAYERS = 2
    D_MODEL = 512
    NUM_HEADS = 8
    DFF = 2048
    INPUT_VOCAB_SIZE = 10000  # 示例输入词汇表大小
    TARGET_VOCAB_SIZE = 10000  # 示例目标词汇表大小
    PE_INPUT = 1000  # 输入序列的最大位置编码长度
    PE_TARGET = 1000  # 目标序列的最大位置编码长度
    BATCH_SIZE = 32

    # 初始化模型
    model = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=INPUT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
        pe_input=PE_INPUT,
        pe_target=PE_TARGET
    )

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    # 初始化数据集和数据加载器
    test_dataset = ExcelDataset(test_data_path, sheet_name='Sheet1', transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_predictions = []
    all_labels = []

    # 进行预测
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_dataloader):
            outputs, _ = model(features, features, None, None)
            outputs = outputs[:, 0, :]  # 取每个序列的第一个输出
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            print(f"Batch {batch_idx}: Predicted {predicted.shape[0]} items")

    # 计算指标
    acc = accuracy_score(all_labels, all_predictions)

    # 对于多分类问题，使用one-vs-rest策略计算AUC
    try:
        auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr')
    except ValueError:
        auc = "无法计算，因为标签只有一种类别"

    cm = confusion_matrix(all_labels, all_predictions)

    # 计算每个类别的精确率和召回率
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)

    # 打印指标
    print(f"Accuracy: {acc}")
    print(f"AUC: {auc}")

    # 打印每个类别的精确率和召回率
    for i, (prec, rec) in enumerate(zip(precision, recall)):
        print(f"Class {i} - Precision: {prec}, Recall: {rec}")

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(cm)

    # 将预测值转换为字符串
    predicted_labels = ['periodontal' if pred == 1 else 'normal' for pred in all_predictions]

    # 创建一个新的DataFrame来保存预测结果
    predictions_df = pd.DataFrame(predicted_labels, columns=['Predicted'])

    # 写入新的xlsx文件
    predictions_df.to_excel(output_path, index=False, header=True)  # 写入标题行

    print(f"预测完成，结果已写入{output_path}")

# 使用测试模块
if __name__ == "__main__":
    model_path = 'transformer_model.pth'
    test_data_path = 'E:/D/1.xlsx'  # 测试集文件路径
    output_path = 'predicted_dataset.xlsx'  # 输出文件路径
    test_model(model_path, test_data_path, output_path)

