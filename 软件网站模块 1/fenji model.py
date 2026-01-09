import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 使用GRU替代简单RNN，增加模型表达能力
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # 只在多层GRU时添加dropout
            bidirectional=False  # 可扩展为双向
        )

        # 添加BatchNorm层提高训练稳定性
        self.bn = nn.BatchNorm1d(hidden_size)

        # 添加额外的全连接层增加模型复杂度
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # 添加dropout层防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 自定义权重初始化
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # 确保输入是float32类型
        if x.dtype != torch.float32:
            x = x.float()

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU前向传播
        out, _ = self.gru(x, h0)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]

        # 应用BatchNorm (需要调整维度以适应BatchNorm要求)
        out_bn = self.bn(out)

        # 通过第一个全连接层并应用激活函数
        out_fc1 = self.fc1(out_bn)
        out_relu = F.relu(out_fc1)

        # 应用dropout
        out_dropout = self.dropout(out_relu)

        # 通过最终分类层
        out = self.fc2(out_dropout)

        return out


# 参数设置
input_size = 32  # 输入特征的维度
hidden_size = 50  # 隐藏状态的维度
num_layers = 3  # 减少层数防止梯度消失/爆炸
num_classes = 3  # 分类任务的类别数
dropout = 0.2  # Dropout概率

# 实例化升级后的模型
upgraded_model = RNNModule(input_size, hidden_size, num_layers, num_classes, dropout)



