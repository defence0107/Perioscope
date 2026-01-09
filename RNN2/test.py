import torch
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import ExcelDataset
from model import RNNModule
from train import RNNClassifierTrainer
import os


def test_model(config):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    test_dataset = ExcelDataset(
        excel_file=config['excel_file'],
        sheet_name=config['sheet_name'],
        transform=None
    )

    # 自动获取输入特征维度
    input_size = test_dataset[0][0].shape[0]

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False  # 测试集通常不需要shuffle
    )

    # 初始化模型
    model = RNNModule(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    )

    # 加载训练好的权重
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 初始化训练器
    trainer = RNNClassifierTrainer(
        model=model,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        device=device
    )

    # 创建结果存储结构
    results = []
    columns = [
        'Timestamp', 'Loss', 'Accuracy', 'AUC',
        'Precision', 'Recall', 'F1', 'Specificity'
    ]

    # 执行单次测试
    print("\n开始测试...")
    with torch.no_grad():  # 禁用梯度计算
        test_metrics = trainer.validate(test_loader, mode='test')

    # 记录当前时间
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建结果条目
    record = {
        'Timestamp': timestamp,
        'Loss': test_metrics[0],
        'Accuracy': test_metrics[1],
        'AUC': test_metrics[2],
        'Precision': test_metrics[3],
        'Recall': test_metrics[4],
        'F1': test_metrics[5],
        'Specificity': test_metrics[6]
    }
    results.append(record)

    # 转换为DataFrame并保存
    df = pd.DataFrame(results, columns=columns)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)

    # 写入Excel文件
    df.to_excel(config['output_path'], index=False)
    print(f"\n测试结果已保存至：{config['output_path']}")

    # 显示最终指标
    print("\n测试指标：")
    print(f"损失值: {test_metrics[0]:.4f}")
    print(f"准确率: {test_metrics[1]:.2f}%")
    print(f"AUC: {test_metrics[2]:.4f}")
    print(f"精确率: {test_metrics[3]:.4f}")
    print(f"召回率: {test_metrics[4]:.4f}")
    print(f"F1分数: {test_metrics[5]:.4f}")
    print(f"特异性: {test_metrics[6]:.4f}")


if __name__ == "__main__":
    # 配置参数
    config = {
        'excel_file': 'E:/D/1.xlsx',
        'sheet_name': 'Sheet1',
        'batch_size': 1,
        'hidden_size': 50,
        'num_layers': 4,
        'num_classes': 2,
        'model_path': 'E:/D/RNN/saved_models/latest_model.pth',
        'output_path': 'E:/D/RNN/test_results/test_metrics.xlsx'
    }

    # 运行测试
    test_model(config)