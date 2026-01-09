import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # 双向RNN

        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 双向LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 修正BatchNorm层的输入维度
        self.bn1 = nn.BatchNorm1d(hidden_size * self.num_directions)  # 双向输出维度
        self.bn2 = nn.BatchNorm1d(hidden_size)  # 与fc2输出维度匹配

        # 调整全连接层维度，确保各层之间维度匹配
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # 输出维度与bn2期望的输入维度匹配
        self.fc3 = nn.Linear(hidden_size, num_classes)

        # 注意力机制层
        self.attention = nn.Linear(hidden_size * self.num_directions, 1)

        # 分层的dropout设置
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 1.5)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.attention.bias)

    def forward(self, x):
        # 确保输入是float32类型
        if x.dtype != torch.float32:
            x = x.float()

        # 输入投影和非线性变换
        x = self.input_proj(x)
        x = F.relu(x)

        # 初始化LSTM的隐藏状态和细胞状态
        device = x.device
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size*num_directions)

        # 应用注意力机制
        attn_weights = F.softmax(self.attention(out).squeeze(-1), dim=1)  # (batch_size, seq_len)
        out = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)  # (batch_size, hidden_size*num_directions)

        # 第一个BatchNorm和dropout
        out = self.bn1(out)
        out = self.dropout1(out)

        # 第一个全连接层
        out = self.fc1(out)  # 输出维度: hidden_size * 2
        out = F.relu(out)

        # 第二个全连接层 - 调整维度以匹配bn2的输入要求
        out = self.fc2(out)  # 输出维度: hidden_size
        out = F.elu(out)

        # 第二个BatchNorm和dropout - 现在输入维度与bn2期望的一致
        out = self.bn2(out)
        out = self.dropout2(out)

        # 最终分类层
        out = self.fc3(out)

        return out


# 参数设置
input_size = 32
hidden_size = 64  # 隐藏层维度
num_layers = 4
num_classes = 3
dropout = 0.3

# 实例化修复后的模型
fixed_rnn_model = RNNModule(input_size, hidden_size, num_layers, num_classes, dropout)




