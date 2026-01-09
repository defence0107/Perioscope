import torch
import torch.nn as nn
import torch.optim as optim
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_attention = nn.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 调用scaled_dot_product_attention函数
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = k.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].clone().detach()

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, trg_mask, src_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, trg_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, src_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_feature_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # 使用线性层替代Embedding处理连续特征
        self.feature_proj = nn.Linear(input_feature_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask):
        # 输入x形状: (batch_size, seq_len=1, feature_size)
        x = self.feature_proj(x)  # 投影到d_model维度
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, trg_mask, src_mask):
        seq_len = x.size(1)
        attention_weights = {}
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, trg_mask, src_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        return x, attention_weights

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_feature_size, num_classes, pe_input, rate=0.1):
        super().__init__()
        # 仅使用编码器部分
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_feature_size, pe_input, rate)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, src_mask=None):
        enc_output = self.encoder(src, src_mask)
        # 取序列第一个位置的输出（当seq_len=1时）
        pooled = enc_output.mean(dim=1)  # 或者使用enc_output[:, 0, :]
        return self.fc(pooled)