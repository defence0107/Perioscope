import pandas as pd
import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import warnings
from model2 import resnet18_3d  # 确保model.py在相同目录或正确路径

warnings.filterwarnings('ignore')

# 特征顺序：5个牙周临床特征 + 12个人口学特征（共17个输入特征，匹配模型输入维度）
COLUMN_ORDER = [
    # 牙周临床特征（5个，含CT推理的inference_code）
    'Periodontal pocket',  # 1. 牙周袋深度(mm)
    'CAL',  # 2. 临床附着丧失(mm)
    'Looseness',  # 4. 牙齿松动度(0-3级)
    'inference_code',  # 5. CT影像推理编码(1-3)

    # 人口学特征（12个，均为数值型）
    '姓名',  # 患者姓名（非特征，仅用于标识）
    'age',  # 年龄(岁)
    'weight',  # 体重(kg)
    '性别',  # 性别(1=男,2=女)
    'Educational level',  # 学历(1-4级)
    'Smoke',  # 吸烟状态(0=不吸,1=吸)
    'Smoking frequency',  # 吸烟频率(0=不吸,1-5级)
    'Degree of smoking',  # 吸烟程度(0=不吸,1-5级)
    '饮酒状态',  # 饮酒状态(0=不饮,1=饮)
    '饮酒频率',  # 饮酒频率(0=不饮,1-5级)
    '家庭年收入',  # 家庭年收入(1-3级)
    'healthy diet',  # 健康饮食(1-3级)
    'trouble sleep',  # 睡眠障碍(0=无,1=有)
    '牙周炎严重程度'  # 预测结果（I-IV期）
]


def collect_patient_data():
    """收集患者完整数据（17个特征），含输入验证"""
    patients = []
    while True:
        patient_name = input("请输入患者姓名（输入q退出）：")
        if patient_name.lower() == 'q':
            break
        if not patient_name.strip():
            print("错误：患者姓名不能为空，请重新输入。")
            continue

        # -------------------------- 1. 牙周临床特征（5个）--------------------------
        # 1.1 牙周袋深度（正数）
        while True:
            try:
                periodontal_pocket = float(input("请输入牙周袋深度（单位：mm）："))
                if periodontal_pocket > 0:
                    break
                print("错误：牙周袋深度必须为正数，请重新输入。")
            except ValueError:
                print("错误：请输入有效的数字。")

        # 1.2 临床附着丧失（非负数）
        while True:
            try:
                cal = float(input("请输入临床附着丧失（单位：mm）："))
                if cal >= 0:
                    break
                print("错误：临床附着丧失不能为负数，请重新输入。")
            except ValueError:
                print("错误：请输入有效的数字。")

        # 1.4 牙齿松动度（0-3）
        while True:
            looseness = input("请输入最大牙齿松动度（0=None，1=I度，2=II度，3=III度）：")
            if looseness in ['0', '1', '2', '3']:
                looseness = int(looseness)
                break
            print("错误：请输入0-3之间的数字。")

        # 1.5 inference_code（CT推理或手动输入，1-3）
        while True:
            print("\ninference_code获取方式：")
            print("1. 手动输入（1-3之间的数字）")
            print("2. 通过CT影像自动推理")
            choice = input("请选择（1/2）：")
            if choice == '1':
                try:
                    inference_code = int(input("请输入inference_code（1-3）："))
                    if 1 <= inference_code <= 3:
                        break
                    print("错误：inference_code必须是1-3之间的数字。")
                except ValueError:
                    print("错误：请输入有效的整数。")
            elif choice == '2':
                inference_code = run_ct_inference_pipeline()
                if inference_code is not None and 1 <= inference_code <= 3:
                    print(f"CT推理成功，inference_code={inference_code}")
                    break
                print("CT推理失败，请重试或选择手动输入。")
            else:
                print("错误：请输入1或2。")

        # -------------------------- 2. 人口学特征（12个）--------------------------
        # 2.1 年龄（正整数）
        while True:
            try:
                age = int(input("请输入患者年龄（岁）："))
                if age > 0:
                    break
                print("错误：年龄必须为正整数，请重新输入。")
            except ValueError:
                print("错误：请输入有效的整数。")

        # 2.2 体重（正数）
        while True:
            try:
                weight = float(input("请输入患者体重（kg）："))
                if weight > 0:
                    break
                print("错误：体重必须为正数，请重新输入。")
            except ValueError:
                print("错误：请输入有效的数字。")

        # 2.3 性别（1=男，2=女）
        while True:
            gender = input("请输入患者性别（1=男，2=女）：")
            if gender in ['1', '2']:
                gender = int(gender)
                break
            print("错误：请输入1（男）或2（女）。")

        # 2.4 学历（1-4级）
        while True:
            edu = input("请输入学历（1=小学及以下，2=初中/中专，3=高中，4=本科及以上）：")
            if edu in ['1', '2', '3', '4']:
                education = int(edu)
                break
            print("错误：请输入1-4之间的数字。")

        # 2.5 吸烟状态（0=不吸，1=吸）
        while True:
            smoke = input("是否吸烟（1=是，0=否）：")
            if smoke in ['0', '1']:
                smoke_status = int(smoke)
                break
            print("错误：请输入0或1。")

        # 2.6 吸烟频率（0=不吸，1-5级）
        if smoke_status == 1:
            while True:
                sf = input("吸烟频率（1=偶尔，2=月1-2次，3=半月1-2次，4=周4-6次，5=每天）：")
                if sf in ['1', '2', '3', '4', '5']:
                    smoke_fre = int(sf)
                    break
                print("错误：请输入1-5之间的数字。")
        else:
            smoke_fre = 0

        # 2.7 吸烟程度（0=不吸，1-5级）
        if smoke_status == 1:
            while True:
                sd = input("每天吸烟量（1=<5支，2=5-10支，3=11-20支，4=21-30支，5=>30支）：")
                if sd in ['1', '2', '3', '4', '5']:
                    smoke_dre = int(sd)
                    break
                print("错误：请输入1-5之间的数字。")
        else:
            smoke_dre = 0

        # 2.8 饮酒状态（0=不饮，1=饮）
        while True:
            drink = input("是否饮酒（1=是，0=否）：")
            if drink in ['0', '1']:
                drink_status = int(drink)
                break
            print("错误：请输入0或1。")

        # 2.9 饮酒频率（0=不饮，1-5级）
        if drink_status == 1:
            while True:
                df = input("饮酒频率（1=偶尔，2=月1-2次，3=半月1-2次，4=周4-6次，5=每天）：")
                if df in ['1', '2', '3', '4', '5']:
                    drink_fre = int(df)
                    break
                print("错误：请输入1-5之间的数字。")
        else:
            drink_fre = 0

        # 2.10 家庭年收入（1-3级）
        while True:
            income = input("家庭年收入（1=≤10万，2=10-20万，3=≥20万）：")
            if income in ['1', '2', '3']:
                family_income = int(income)
                break
            print("错误：请输入1-3之间的数字。")

        # 2.11 健康饮食（1-3级）
        while True:
            diet = input("用餐规律（1=非常规律，2=有时规律，3=完全不规律）：")
            if diet in ['1', '2', '3']:
                healthy = int(diet)
                break
            print("错误：请输入1-3之间的数字。")

        # 2.12 睡眠障碍（0=无，1=有）
        while True:
            sleep = input("是否有睡眠障碍（1=是，0=否）：")
            if sleep in ['0', '1']:
                trouble_sleep = int(sleep)
                break
            print("错误：请输入0或1。")

        # 组织数据（按17个特征顺序）
        patients.append({
            # 牙周临床特征（5个）
            'Periodontal pocket': periodontal_pocket,
            'CAL': cal,
            'Looseness': looseness,
            'inference_code': inference_code,
            # 人口学特征（12个）
            '姓名': patient_name,
            'age': age,
            'weight': weight,
            '性别': gender,
            'Educational level': education,
            'Smoke': smoke_status,
            'Smoking frequency': smoke_fre,
            'Degree of smoking': smoke_dre,
            '饮酒状态': drink_status,
            '饮酒频率': drink_fre,
            '家庭年收入': family_income,
            'healthy diet': healthy,
            'trouble sleep': trouble_sleep
        })
        print(f"\n✅ 患者「{patient_name}」数据录入完成\n")

    return patients


def save_to_excel(patients, file_path="patient_periodontal_data1.xlsx"):
    """保存/追加患者数据到Excel，确保17个特征列完整"""
    if not patients:
        print("❌ 无数据可保存")
        return

    # 构建DataFrame，确保列顺序与COLUMN_ORDER一致
    df_new = pd.DataFrame(patients, columns=COLUMN_ORDER)

    # 处理文件追加逻辑
    if os.path.exists(file_path):
        try:
            df_exist = pd.read_excel(file_path)
            # 补全缺失列（避免旧数据缺少新特征）
            for col in COLUMN_ORDER:
                if col not in df_exist.columns:
                    df_exist[col] = None
            # 对齐列顺序
            df_exist = df_exist[COLUMN_ORDER]
            # 合并数据
            df_combined = pd.concat([df_exist, df_new], ignore_index=True)
            df_combined.to_excel(file_path, index=False)
            print(f"✅ 数据已追加到：{file_path}")
        except Exception as e:
            print(f"❌ 追加数据失败：{str(e)}")
    else:
        try:
            df_new.to_excel(file_path, index=False)
            print(f"✅ 新文件已创建：{file_path}")
        except Exception as e:
            print(f"❌ 创建文件失败：{str(e)}")


# ========== 完全匹配权重文件的模型（维度+键名双对齐） ==========
class PeriodontalLSTMClassifier(torch.nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=4, num_classes=4):
        super(PeriodontalLSTMClassifier, self).__init__()
        # 1. 输入归一化层（匹配权重：input_norm.weight [17]）
        self.input_norm = torch.nn.BatchNorm1d(input_size)

        # 2. 特征提取层（匹配权重：feature_extractor.0.weight [128,17]）
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),  # 17→128
            torch.nn.ReLU()
        )

        # 3. LSTM层（匹配权重：双向，4层，hidden_size=128）
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 输出维度=128×2=256
        )

        # 4. 位置编码（匹配权重：pos_encoder.pe [5000,1,256]）
        self.pos_encoder = torch.nn.Module()
        self.pos_encoder.pe = torch.nn.Parameter(
            torch.zeros(5000, 1, hidden_size * 2),  # 形状严格匹配权重：5000×1×256
            requires_grad=False  # 固定位置编码，不参与训练
        )

        # 5. Transformer层（匹配权重键名+维度：feed_forward用1024维）
        self.transformer = torch.nn.Module()
        # 5.1 第一层归一化（匹配：transformer.norm1.*）
        self.transformer.norm1 = torch.nn.LayerNorm(hidden_size * 2)  # 输入256维
        # 5.2 注意力层（匹配：transformer.attention.query/key/value/out.*）
        self.transformer.attention = torch.nn.Module()
        self.transformer.attention.query = torch.nn.Linear(hidden_size * 2, hidden_size * 2)  # 256→256
        self.transformer.attention.key = torch.nn.Linear(hidden_size * 2, hidden_size * 2)  # 256→256
        self.transformer.attention.value = torch.nn.Linear(hidden_size * 2, hidden_size * 2)  # 256→256
        self.transformer.attention.out = torch.nn.Linear(hidden_size * 2, hidden_size * 2)  # 256→256
        # 5.3 前馈网络（匹配权重：feed_forward.linear1 [1024,256]，linear2 [256,1024]）
        self.transformer.feed_forward = torch.nn.Module()
        self.transformer.feed_forward.linear1 = torch.nn.Linear(hidden_size * 2, 1024)  # 256→1024（关键调整）
        self.transformer.feed_forward.linear2 = torch.nn.Linear(1024, hidden_size * 2)  # 1024→256（关键调整）
        # 5.4 第二层归一化（匹配：transformer.norm2.*）
        self.transformer.norm2 = torch.nn.LayerNorm(hidden_size * 2)  # 输入256维

        # 6. 注意力池化层（匹配权重：attention.*）
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),  # 256→128
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)  # 128→1（权重）
        )

        # 7. 输出层（匹配权重：output.0 [512,256]，output.3 [256,512]，output.6 [4,256]）
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 512),  # 256→512
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),  # 512→256
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)  # 256→4（4分类：I-IV期）
        )

    def forward(self, x):
        """前向传播：严格匹配权重训练时的计算逻辑"""
        batch_size, seq_len, input_dim = x.shape  # 输入形状：(batch, seq_len=1, 17)

        # 1. 输入归一化（处理维度顺序：适配BatchNorm1d）
        x = x.permute(0, 2, 1).contiguous()  # (batch, 17, 1)
        x = self.input_norm(x)  # 归一化
        x = x.permute(0, 2, 1).contiguous()  # 恢复：(batch, 1, 17)

        # 2. 特征提取（17→128）
        x = self.feature_extractor(x)  # (batch, 1, 128)

        # 3. LSTM层（128→256，双向）
        lstm_out, _ = self.lstm(x)  # (batch, 1, 256)

        # 4. 位置编码（截取对应序列长度，避免超出权重形状）
        if seq_len <= 5000:
            # 取前seq_len个位置的编码（权重形状5000×1×256，适配任意seq_len≤5000）
            pe = self.pos_encoder.pe[:seq_len, :, :].unsqueeze(0)  # (1, seq_len, 1, 256)
            pe = pe.repeat(batch_size, 1, 1, 1).squeeze(2)  # (batch, seq_len, 256)
        else:
            # 若seq_len>5000，循环使用位置编码（实际应用中seq_len=1，此分支备用）
            pe = self.pos_encoder.pe.repeat(seq_len // 5000 + 1, 1, 1)[:seq_len, :, :]
            pe = pe.unsqueeze(0).repeat(batch_size, 1, 1, 1).squeeze(2)
        lstm_out += pe  # 叠加位置编码：(batch, 1, 256)

        # 5. Transformer层（注意力+前馈网络）
        # 5.1 注意力层 + 残差连接 + 归一化
        q = self.transformer.attention.query(lstm_out)  # (batch, 1, 256)
        k = self.transformer.attention.key(lstm_out)  # (batch, 1, 256)
        v = self.transformer.attention.value(lstm_out)  # (batch, 1, 256)
        # 缩放点积注意力计算
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])  # (batch, 1, 1)
        attn_weight = torch.softmax(attn_score, dim=-1)  # (batch, 1, 1)
        attn_out = torch.matmul(attn_weight, v)  # (batch, 1, 256)
        attn_out = self.transformer.attention.out(attn_out)  # (batch, 1, 256)
        # 残差连接 + 归一化
        norm1_out = self.transformer.norm1(lstm_out + attn_out)  # (batch, 1, 256)

        # 5.2 前馈网络 + 残差连接 + 归一化（1024维中间层）
        ff_out = self.transformer.feed_forward.linear1(norm1_out)  # (batch, 1, 256)→(batch, 1, 1024)
        ff_out = torch.nn.functional.relu(ff_out)  # 激活函数
        ff_out = self.transformer.feed_forward.linear2(ff_out)  # (batch, 1, 1024)→(batch, 1, 256)
        # 残差连接 + 归一化
        transformer_out = self.transformer.norm2(norm1_out + ff_out)  # (batch, 1, 256)

        # 6. 注意力池化（全局池化，适配任意序列长度）
        attn_weights = self.attention(transformer_out)  # (batch, 1, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # 权重归一化
        pooled_out = torch.sum(transformer_out * attn_weights, dim=1)  # (batch, 256)

        # 7. 输出层（256→512→256→4）
        final_out = self.output(pooled_out)  # (batch, 4)
        return final_out


def load_and_predict(file_path="patient_periodontal_data1.xlsx",
                     model_path="E:/D/软件网站模块/fenqi.pth"):
    try:
        # 1. 加载数据（16个特征）
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"患者数据文件不存在：{file_path}")
        df = pd.read_excel(file_path)
        print(f"✅ 成功加载 {len(df)} 条患者数据")

        # 2. 提取16个输入特征（移除冗余特征）
        feature_cols = [
            'Periodontal pocket', 'CAL', 'Looseness', 'inference_code',
            'age', 'weight', '性别', 'Educational level', 'Smoke',
            'Smoking frequency', 'Degree of smoking', '饮酒状态', '饮酒频率',
            '家庭年收入', 'healthy diet', 'trouble sleep'
        ]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要特征列：{', '.join(missing_cols)}")

        X = df[feature_cols].copy()
        X = X.fillna(0).astype(np.float32)  # 缺失值填充

        # 3. 加载模型（16维输入）
        model = PeriodontalLSTMClassifier(input_size=16)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 4. 加载权重并裁剪
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}

        for key in state_dict:
            weight = state_dict[key]
            if key == 'feature_extractor.0.weight':  # 裁剪第一层权重（17→16）
                weight = weight[:, :16]  # 保留前16列（假设冗余特征是最后一列）
            elif key == 'input_norm.weight' or key == 'input_norm.bias':
                weight = weight[:16]  # 裁剪BatchNorm参数
            elif key == 'input_norm.running_mean' or key == 'input_norm.running_var':
                weight = weight[:16]
            new_state_dict[key] = weight

        model.load_state_dict(new_state_dict, strict=False)  # 允许部分权重不匹配
        model.to(device).eval()
        print(f"✅ 模型权重加载成功（已裁剪为16维输入）")

        # 5. 预测
        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)  # 计算概率
            _, y_pred = torch.max(outputs, dim=1)

        # 6. 打印预测结果和概率
        severity_map = {0: "I期", 1: "II期", 2: "III期", 3: "IV期"}
        print("\n📋 预测结果概览：")
        print("-" * 60)
        print(f"{'患者索引':<10}{'预测结果':<10}{'I期概率':<10}{'II期概率':<10}{'III期概率':<10}{'IV期概率':<10}")
        print("-" * 60)

        for i in range(len(df)):
            probs_list = [f"{p:.2%}" for p in probs[i].cpu().numpy()]
            print(
                f"{i:<10}{severity_map[y_pred[i].item()]:<10}{probs_list[0]:<10}{probs_list[1]:<10}{probs_list[2]:<10}{probs_list[3]:<10}")

        # 7. 保存结果
        df['牙周炎严重程度'] = [severity_map[p] for p in y_pred.cpu().numpy()]
        # 添加概率列
        for i, stage in enumerate(['I期概率', 'II期概率', 'III期概率', 'IV期概率']):
            df[stage] = probs[:, i].cpu().numpy()

        output_path = file_path.replace('.xlsx', '_prediction.xlsx')
        df.to_excel(output_path, index=False)
        print("\n✅ 预测结果已保存至:", output_path)

    except Exception as e:
        print(f"❌ 错误：{str(e)}")


class CTInferencePipeline:
    """CT影像推理流水线：读取nii文件→加载3D ResNet→输出inference_code（1-3）"""

    def __init__(self, nii_path, model_weight_path):
        self.nii_path = Path(nii_path)
        self.model_path = Path(model_weight_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.nii_data = None

    def load_nii(self):
        """读取nii文件并预处理（添加通道和批次维度）"""
        try:
            if not self.nii_path.exists():
                raise FileNotFoundError(f"CT文件不存在：{self.nii_path}")
            if self.nii_path.suffix not in ['.nii', '.nii.gz']:
                raise ValueError(f"文件格式错误，需为.nii或.nii.gz：{self.nii_path.suffix}")

            # 读取nii数据
            nii_img = nib.load(str(self.nii_path))
            self.nii_data = nii_img.get_fdata()  # 形状：(H, W, D)
            # 预处理：添加通道维度（1）和批次维度（1），匹配3D ResNet输入
            self.nii_data = np.expand_dims(self.nii_data, axis=0)  # (1, H, W, D)
            self.nii_data = np.expand_dims(self.nii_data, axis=0)  # (1, 1, H, W, D)
            self.nii_data = torch.tensor(self.nii_data, dtype=torch.float32)

            print(f"✅ CT文件读取成功：{self.nii_path.name}")
            print(f"   数据形状：{self.nii_data.shape}")
            return True
        except Exception as e:
            print(f"❌ CT文件读取失败：{str(e)}")
            return False

    def load_ct_model(self):
        """加载3D ResNet模型和权重"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"CT模型权重不存在：{self.model_path}")
            if self.model_path.suffix != '.pth':
                raise ValueError(f"权重格式错误，需为.pth：{self.model_path.suffix}")

            # 加载3D ResNet模型（num_classes=3，对应inference_code 1-3）
            self.model = resnet18_3d(num_classes=3, in_channels=1).to(self.device)
            # 加载权重（处理多GPU前缀）
            state_dict = torch.load(str(self.model_path), map_location=self.device, weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k[7:] if k.startswith('module.') else k
                new_state_dict[new_key] = v
            # 非严格加载（允许部分无关权重不匹配）
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()  # 评估模式

            print(f"✅ CT模型加载成功，运行设备：{self.device}")
            return True
        except Exception as e:
            print(f"❌ CT模型加载失败：{str(e)}")
            return False

    def run_ct_inference(self):
        """执行CT推理，返回inference_code（1-3）"""
        try:
            if self.nii_data is None:
                raise ValueError("请先调用load_nii()加载CT数据")
            if self.model is None:
                raise ValueError("请先调用load_ct_model()加载CT模型")

            # 推理计算
            with torch.no_grad():
                input_data = self.nii_data.to(self.device)
                logits = self.model(input_data)  # (1, 3)
                probs = torch.softmax(logits, dim=1)  # 概率归一化
                pred_class = torch.argmax(probs, dim=1).item()  # 0-2
                inference_code = pred_class + 1  # 转换为1-3

            print(f"\n✅ CT推理完成")
            print(f"   类别概率：{probs.cpu().numpy()[0].round(4)}")
            print(f"   inference_code：{inference_code}")
            return inference_code
        except Exception as e:
            print(f"❌ CT推理失败：{str(e)}")
            return None


def run_ct_inference_pipeline():
    """外部调用接口：配置CT推理参数并执行"""
    # 配置CT相关路径（根据实际情况修改）
    CT_NII_PATH = "E:/D/data_set/raw_dataset/test/CT/volume-0.nii"
    CT_MODEL_PATH = "E:/D/软件网站模块/fenlei.pth"

    print("\n" + "=" * 50)
    print("           CT影像推理流水线启动")
    print("=" * 50)
    # 初始化流水线
    pipeline = CTInferencePipeline(
        nii_path=CT_NII_PATH,
        model_weight_path=CT_MODEL_PATH
    )
    # 分步执行
    if not pipeline.load_nii():
        return None
    if not pipeline.load_ct_model():
        return None
    return pipeline.run_ct_inference()


if __name__ == "__main__":
    """系统入口：提供数据录入、预测、CT推理功能"""
    print("=" * 60)
    print("        患者牙周炎数据录入与预测系统")
    print("=" * 60)
    while True:
        print("\n请选择操作：")
        print("1. 录入新患者数据")
        print("2. 基于已有数据预测牙周炎严重程度")
        print("3. 单独运行CT影像推理（获取inference_code）")
        print("4. 退出系统")
        choice = input("\n请输入选项（1-4）：")

        if choice == '1':
            print("\n" + "-" * 30)
            print("   新患者数据录入")
            print("-" * 30)
            patient_data = collect_patient_data()
            if patient_data:
                save_to_excel(patient_data)
        elif choice == '2':
            print("\n" + "-" * 30)
            print("   牙周炎严重程度预测")
            print("-" * 30)
            load_and_predict()
        elif choice == '3':
            print("\n" + "-" * 30)
            print("   CT影像推理")
            print("-" * 30)
            code = run_ct_inference_pipeline()
            if code is not None:
                print(f"\n✅ CT推理结果：inference_code = {code}")
            else:
                print("\n❌ CT推理失败")
        elif choice == '4':
            print("\n✅ 系统已退出，感谢使用！")
            break
        else:
            print("\n❌ 无效选项，请输入1-4之间的数字。")


