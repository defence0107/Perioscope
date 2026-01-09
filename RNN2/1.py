import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ExcelDataset
from model import RNNModule
from train import RNNClassifierTrainer
import os

def run_training(config):  # 修改函数名避免test_前缀
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
        shuffle=False
    )

    # 初始化模型
    model = RNNModule(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    )

    # 加载预训练权重
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # 初始化训练器
    trainer = RNNClassifierTrainer(
        model=model,
        train_loader=test_loader,
        val_loader=None,
        test_loader=test_loader,
        device=device
    )

    # 创建结果存储结构
    results = []
    columns = [
        'Epoch', 'Loss', 'Accuracy', 'AUC',
        'Precision', 'Recall', 'F1', 'Specificity'
    ]

    print("\n开始迭代训练...")
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        total_loss = 0.0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # 计算平均损失
        epoch_loss = total_loss / len(test_dataset)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            test_metrics = trainer.validate(test_loader, mode='test')

        # 构建结果条目
        record = {
            'Epoch': epoch + 1,
            'Loss': epoch_loss,
            'Accuracy': test_metrics[1],
            'AUC': test_metrics[2],
            'Precision': test_metrics[3],
            'Recall': test_metrics[4],
            'F1': test_metrics[5],
            'Specificity': test_metrics[6]
        }
        results.append(record)

        # 打印epoch指标
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] - "
              f"Loss: {epoch_loss:.4f} | "
              f"Acc: {test_metrics[1]:.2f}% | "
              f"AUC: {test_metrics[2]:.4f} | "
              f"Precision: {test_metrics[3]:.4f} | "
              f"Recall: {test_metrics[4]:.4f} | "
              f"F1: {test_metrics[5]:.4f} | "
              f"Specificity: {test_metrics[6]:.4f}")

    # 保存结果到Excel
    df = pd.DataFrame(results, columns=columns)
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    df.to_excel(config['output_path'], index=False)
    print(f"\n完整训练日志已保存至：{config['output_path']}")

if __name__ == "__main__":
    # 更新配置参数
    config = {
        'excel_file': 'E:/D/1.xlsx',
        'sheet_name': 'Sheet1',
        'batch_size': 4,
        'hidden_size': 50,
        'num_layers': 4,
        'num_classes': 2,
        'model_path': 'E:/D/RNN/saved_models/latest_model.pth',
        'output_path': 'E:/D/RNN/test_results/training_log.xlsx',
        'num_epochs': 100,
        'learning_rate': 0.001
    }

    # 直接运行训练程序
    run_training(config)  # 修改函数调用名称