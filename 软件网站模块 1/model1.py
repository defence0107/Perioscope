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

        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

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
        attention_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class LSTM(nn.Module):
    # 关键修改1：默认启用Transformer（use_transformer=True），避免权重中Transformer参数多余
    def __init__(self, input_size=17, hidden_size=128, num_layers=4, num_classes=4,
                 dropout=0.3, bidirectional=True, use_attention=True, use_transformer=True):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_transformer = use_transformer

        # 关键修改2：input_size=17（匹配权重中[17]形状的输入层参数）
        self.input_norm = nn.BatchNorm1d(input_size)

        # 特征提取层：输入维度=17（匹配权重中[128,17]的Linear层参数）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 双向LSTM（输入维度=hidden_size=128，与特征提取层输出一致）
        lstm_dim = hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # LSTM输出维度（双向→2*hidden_size）
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        # 保留Transformer模块（与权重中的pos_encoder/transformer参数对应）
        if use_transformer:
            self.pos_encoder = PositionalEncoding(lstm_out_dim)
            self.transformer = TransformerBlock(
                hidden_size=lstm_out_dim,
                num_heads=4,
                ff_dim=lstm_out_dim * 4,
                dropout=dropout
            )

        # 注意力机制（权重中无冲突，保持不变）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_out_dim // 2, 1)
            )

        # 输出层：若权重中num_classes=2，需确保此处num_classes=2（默认已设为2，可根据权重调整）
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
        # 输入形状需为 [batch_size, seq_len, input_size=17]（与权重匹配）
        batch_size, seq_len, input_size = x.size()

        # 输入层归一化（维度适配BatchNorm1d：[batch_size*seq_len, 17]）
        x_reshaped = x.contiguous().view(-1, input_size)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, input_size)

        # 特征提取（输出维度：[batch_size, seq_len, 128]）
        x = self.feature_extractor(x)

        # 初始化LSTM隐藏状态（设备自适应，避免CPU/GPU不匹配）
        device = x.device
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size, device=device)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 启用Transformer（与权重中的Transformer参数对应，避免多余参数报错）
        if self.use_transformer:
            lstm_out = self.pos_encoder(lstm_out)
            lstm_out = self.transformer(lstm_out)

        # 注意力机制
        if self.use_attention:
            attention_weights = self.attention(lstm_out)
            attention_weights = F.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)
        else:
            context = lstm_out[:, -1, :]

        # 输出层
        output = self.output(context)

        return output


# ------------------- 模型使用示例（关键配置已适配权重） -------------------
if __name__ == "__main__":
    # 1. 加载模型（无需修改参数，默认input_size=17、use_transformer=True）
    model = LSTM(
        num_classes=4 # 关键：若权重训练时是2分类则保持2，若为其他类别需修改（如4分类改为4）
    )

    # 2. 模拟加载权重（替换为你的权重路径，此时无报错）
    model_path = "你的权重文件路径.pth"  # 例如："periodontal_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("权重加载成功！")
    except Exception as e:
        print(f"加载失败：{e}")

    # 3. 测试输入（需为17维特征，与input_size=17匹配）
    batch_size = 1
    seq_len = 5  # 序列长度需与训练权重时一致（若未知可先设为5测试）
    dummy_input = torch.randn(batch_size, seq_len, 17)  # 输入维度：[1,5,17]
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
