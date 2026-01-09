import pandas as pd
import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import warnings
from model2 import resnet18_3d  # 确保model.py在相同目录或正确路径

warnings.filterwarnings('ignore')

# 特征顺序：5个牙周临床特征 + 15个人口学特征（共20个输入特征，匹配模型输入维度）
COLUMN_ORDER = [
    # 牙周临床特征（5个，含CT推理的inference_code）
    'Periodontal pocket',  # 1. 牙周袋深度(mm)
    'CAL',  # 2. 临床附着丧失(mm)
    'Looseness',  # 4. 牙齿松动度(0-3级)
    'inference_code',  # 5. CT影像推理编码(1-3)

    # 人口学特征（15个，均为数值型）
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
    'Porphyromonas_endodontalis',  # 新增特征：牙髓卟啉单胞菌
    'Porphyromonas_gingivalis',  # 新增特征：中间密螺旋体
    'Campylobacter_gracilis',  # 新增特征：纤细弯曲菌
    '牙周炎严重程度'  # 预测结果（0=I期，2=III期，3=IV期）
]

# 三分类映射：模型输出索引→临床分期（0=I期，1=III期，2=IV期）
# 注意：模型输出是0/1/2，需要映射为临床标注的0/2/3
SEVERITY_MAP = {
    0: "I期",    # 模型输出0 → 临床I期（标注0）
    1: "III期",  # 模型输出1 → 临床III期（标注2）
    2: "IV期"    # 模型输出2 → 临床IV期（标注3）
}

# 临床分期反向映射（用于结果保存和展示）
CLINICAL_LABEL_MAP = {
    0: "I期",
    2: "III期",
    3: "IV期"
}


def collect_patient_data():
    """收集患者完整数据（20个特征），含输入验证"""
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

        # -------------------------- 2. 人口学特征（15个）--------------------------
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

        # 新增特征：牙髓卟啉单胞菌（Porphyromonas_endodontalis）
        while True:
            try:
                porphyromonas = float(input("请输入牙髓卟啉单胞菌数值："))
                break
            except ValueError:
                print("错误：请输入有效的数字。")

        # 新增特征：中间密螺旋体（Treponema_medium）
        while True:
            try:
                treponema = float(input("请输入中间密螺旋体数值："))
                break
            except ValueError:
                print("错误：请输入有效的数字。")

        # 新增特征：纤细弯曲菌（Campylobacter_gracilis）
        while True:
            try:
                campylobacter = float(input("请输入纤细弯曲菌数值："))
                break
            except ValueError:
                print("错误：请输入有效的数字。")

        # 组织数据（按20个特征顺序）
        patients.append({
            # 牙周临床特征（5个）
            'Periodontal pocket': periodontal_pocket,
            'CAL': cal,
            'Looseness': looseness,
            'inference_code': inference_code,
            # 人口学特征（15个）
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
            'trouble sleep': trouble_sleep,
            'Porphyromonas_endodontalis': porphyromonas,
            'Treponema_medium': treponema,
            'Campylobacter_gracilis': campylobacter
        })
        print(f"\n✅ 患者「{patient_name}」数据录入完成\n")

    return patients


def save_to_excel(patients, file_path="patient_periodontal_data1.xlsx"):
    """保存/追加患者数据到Excel，确保20个特征列完整"""
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


# ========== 三分类模型（0=I期，1=III期，2=IV期） ==========
class PeriodontalLSTMClassifier(torch.nn.Module):
    def __init__(self, input_size=19, hidden_size=128, num_layers=4, num_classes=3):
        super(PeriodontalLSTMClassifier, self).__init__()
        # 1. 输入归一化层（19维输入）
        self.input_norm = torch.nn.BatchNorm1d(input_size)

        # 2. 特征提取层（19→128）
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU()
        )

        # 3. LSTM层（双向，4层，输出256维）
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 4. 位置编码（5000×1×256）
        self.pos_encoder = torch.nn.Module()
        self.pos_encoder.pe = torch.nn.Parameter(
            torch.zeros(5000, 1, hidden_size * 2),
            requires_grad=False
        )

        # 5. Transformer层
        self.transformer = torch.nn.Module()
        self.transformer.norm1 = torch.nn.LayerNorm(hidden_size * 2)
        self.transformer.attention = torch.nn.Module()
        self.transformer.attention.query = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.transformer.attention.key = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.transformer.attention.value = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.transformer.attention.out = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.transformer.feed_forward = torch.nn.Module()
        self.transformer.feed_forward.linear1 = torch.nn.Linear(hidden_size * 2, 1024)
        self.transformer.feed_forward.linear2 = torch.nn.Linear(1024, hidden_size * 2)
        self.transformer.norm2 = torch.nn.LayerNorm(hidden_size * 2)

        # 6. 注意力池化层
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )

        # 7. 输出层（256→512→256→3，三分类）
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)  # 输出维度改为3（三分类）
        )

    def forward(self, x):
        """前向传播：输入19维，输出3类概率"""
        batch_size, seq_len, input_dim = x.shape  # (batch, 1, 19)

        # 1. 输入归一化
        x = x.permute(0, 2, 1).contiguous()  # (batch, 19, 1)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, 1, 19)

        # 2. 特征提取
        x = self.feature_extractor(x)  # (batch, 1, 128)

        # 3. LSTM层
        lstm_out, _ = self.lstm(x)  # (batch, 1, 256)

        # 4. 位置编码
        if seq_len <= 5000:
            pe = self.pos_encoder.pe[:seq_len, :, :].unsqueeze(0)
            pe = pe.repeat(batch_size, 1, 1, 1).squeeze(2)  # (batch, seq_len, 256)
        else:
            pe = self.pos_encoder.pe.repeat(seq_len // 5000 + 1, 1, 1)[:seq_len, :, :]
            pe = pe.unsqueeze(0).repeat(batch_size, 1, 1, 1).squeeze(2)
        lstm_out += pe  # (batch, 1, 256)

        # 5. Transformer层
        q = self.transformer.attention.query(lstm_out)
        k = self.transformer.attention.key(lstm_out)
        v = self.transformer.attention.value(lstm_out)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        attn_weight = torch.softmax(attn_score, dim=-1)
        attn_out = torch.matmul(attn_weight, v)
        attn_out = self.transformer.attention.out(attn_out)
        norm1_out = self.transformer.norm1(lstm_out + attn_out)

        ff_out = self.transformer.feed_forward.linear1(norm1_out)
        ff_out = torch.nn.functional.relu(ff_out)
        ff_out = self.transformer.feed_forward.linear2(ff_out)
        transformer_out = self.transformer.norm2(norm1_out + ff_out)  # (batch, 1, 256)

        # 6. 注意力池化
        attn_weights = self.attention(transformer_out)  # (batch, 1, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled_out = torch.sum(transformer_out * attn_weights, dim=1)  # (batch, 256)

        # 7. 输出层（3类）
        final_out = self.output(pooled_out)  # (batch, 3)
        return final_out


def load_and_predict(file_path="patient_periodontal_data1.xlsx",
                     model_path="E:/D/网站/软件网站模块/fenqijun.pth"):
    try:
        # 1. 加载数据（19个特征）
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"患者数据文件不存在：{file_path}")
        df = pd.read_excel(file_path)
        print(f"✅ 成功加载 {len(df)} 条患者数据")

        # 2. 提取19个输入特征
        feature_cols = [
            'Periodontal pocket', 'CAL', 'Looseness', 'inference_code',
            'age', 'weight', '性别', 'Educational level', 'Smoke',
            'Smoking frequency', 'Degree of smoking', '饮酒状态', '饮酒频率',
            '家庭年收入', 'healthy diet', 'trouble sleep',
            'Porphyromonas_endodontalis', 'Treponema_medium', 'Campylobacter_gracilis'
        ]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要特征列：{', '.join(missing_cols)}")

        X = df[feature_cols].copy()
        X = X.fillna(0).astype(np.float32)  # 缺失值填充

        # 3. 加载三分类模型（19维输入，3类输出）
        model = PeriodontalLSTMClassifier(input_size=19, num_classes=3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 4. 加载权重并调整（适配三分类）
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for key in state_dict:
            weight = state_dict[key]
            # 调整输入层权重（如果原有是16维，扩展到19维）
            if key == 'feature_extractor.0.weight':
                if weight.shape[1] == 16:
                    new_weight = torch.cat([
                        weight,
                        torch.zeros(weight.shape[0], 3, device=weight.device, dtype=weight.dtype)
                    ], dim=1)
                    weight = new_weight
            # 调整BatchNorm参数（16维→19维）
            elif key in ['input_norm.weight', 'input_norm.bias', 'input_norm.running_mean', 'input_norm.running_var']:
                if weight.shape[0] == 16:
                    weight = torch.cat([
                        weight,
                        torch.zeros(3, device=weight.device, dtype=weight.dtype)
                    ], dim=0)
            # 调整输出层权重（如果原有是4类，裁剪/扩展到3类）
            elif key == 'output.6.weight':  # 输出层最后一层权重
                if weight.shape[0] == 4:  # 原有4分类→3分类
                    weight = weight[:3, :]  # 保留前3类权重（或根据实际需求调整）
                elif weight.shape[0] != 3:
                    print(f"警告：输出层权重维度不匹配，当前为{weight.shape[0]}类，将自动调整为3类")
                    weight = weight[:3, :] if weight.shape[0] > 3 else torch.cat([
                        weight,
                        torch.zeros(3 - weight.shape[0], weight.shape[1], device=weight.device, dtype=weight.dtype)
                    ], dim=0)
            elif key == 'output.6.bias':  # 输出层偏置
                if weight.shape[0] == 4:
                    weight = weight[:3]
                elif weight.shape[0] != 3:
                    weight = weight[:3] if weight.shape[0] > 3 else torch.cat([
                        weight,
                        torch.zeros(3 - weight.shape[0], device=weight.device, dtype=weight.dtype)
                    ], dim=0)
            new_state_dict[key] = weight

        model.load_state_dict(new_state_dict, strict=False)
        model.to(device).eval()
        print(f"✅ 三分类模型权重加载成功（19维输入→3类输出），运行设备：{device}")

        # 5. 预测
        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)  # 3类概率
            _, y_pred = torch.max(outputs, dim=1)  # 模型输出索引（0/1/2）

        # 6. 转换为临床分期（0→I期，1→III期，2→IV期）
        clinical_stages = [SEVERITY_MAP[pred.item()] for pred in y_pred]
        clinical_labels = [0 if pred.item() == 0 else 2 if pred.item() == 1 else 3 for pred in y_pred]

        # 7. 打印预测结果
        print("\n📋 预测结果概览（三分类：0=I期，2=III期，3=IV期）：")
        print("-" * 90)
        print(f"{'患者索引':<10}{'预测结果':<10}{'临床标注':<10}{'I期概率':<10}{'III期概率':<10}{'IV期概率':<10}")
        print("-" * 90)

        for i in range(len(df)):
            probs_list = [f"{p:.2%}" for p in probs[i].cpu().numpy()]
            print(
                f"{i:<10}{clinical_stages[i]:<10}{clinical_labels[i]:<10}{probs_list[0]:<10}{probs_list[1]:<10}{probs_list[2]:<10}")

        # 8. 保存结果
        df['牙周炎严重程度'] = clinical_stages
        df['临床标注'] = clinical_labels  # 新增临床标注列（0/I期，2/III期，3/IV期）
        # 添加各类概率列
        df['I期概率'] = probs[:, 0].cpu().numpy()
        df['III期概率'] = probs[:, 1].cpu().numpy()
        df['IV期概率'] = probs[:, 2].cpu().numpy()

        output_path = file_path.replace('.xlsx', '_3class_prediction.xlsx')
        df.to_excel(output_path, index=False)
        print(f"\n✅ 三分类预测结果已保存至:", output_path)

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
    """系统入口：三分类牙周炎预测系统"""
    print("=" * 60)
    print("    患者牙周炎三分类预测系统（0=I期，2=III期，3=IV期）")
    print("=" * 60)
    while True:
        print("\n请选择操作：")
        print("1. 录入新患者数据")
        print("2. 基于已有数据预测牙周炎严重程度（三分类）")
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
            print("   牙周炎三分类预测（0=I期，2=III期，3=IV期）")
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