import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # x: [batch_size, seq_len, hidden_size]
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # q, k, v: [batch_size, num_heads, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        out = self.out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, hidden_size)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, ff_dim=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, ff_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力层
        attention_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attention_output))

        # 前馈网络层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=4, num_classes=2,
                 dropout=0.3, bidirectional=True, use_attention=True, use_transformer=False):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_transformer = use_transformer

        # 输入层归一化
        self.input_norm = nn.BatchNorm1d(input_size)

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 双向LSTM
        lstm_dim = hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出维度
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        # 位置编码（如果使用Transformer）
        if use_transformer:
            self.pos_encoder = PositionalEncoding(lstm_out_dim)
            self.transformer = TransformerBlock(
                hidden_size=lstm_out_dim,
                num_heads=4,
                ff_dim=lstm_out_dim * 4,
                dropout=dropout
            )

        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_out_dim // 2, 1)
            )

        # 输出层
        output_dim = lstm_out_dim
        self.output = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.size()

        # 输入层归一化 (需要调整维度以适应BatchNorm1d)
        x_reshaped = x.contiguous().view(-1, input_size)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, input_size)

        # 特征提取
        x = self.feature_extractor(x)

        # 初始化LSTM隐藏状态
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 使用Transformer增强序列表示
        if self.use_transformer:
            lstm_out = self.pos_encoder(lstm_out)
            lstm_out = self.transformer(lstm_out)

        # 应用注意力机制
        if self.use_attention:
            # 计算注意力权重
            attention_weights = self.attention(lstm_out)
            attention_weights = F.softmax(attention_weights, dim=1)

            # 加权求和
            context = torch.sum(attention_weights * lstm_out, dim=1)
        else:
            # 使用最后一个时间步的输出
            context = lstm_out[:, -1, :]

        # 输出层
        output = self.output(context)

        return output


# 示例使用
if __name__ == "__main__":
    # 超参数
    input_size = 16 # 例如：30个特征
    hidden_size = 128
    num_layers = 4
    num_classes = 4
    dropout = 0.3

    # 创建模型
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=True,
        use_attention=True,
        use_transformer=True
    )

    # 示例输入
    batch_size = 32
    seq_len = 5
    dummy_input = torch.randn(batch_size, seq_len, input_size)

    # 前向传播
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")