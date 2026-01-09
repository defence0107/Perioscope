import pandas as pd
import os
import joblib
import torch
import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
from model2 import resnet18_3d  # 确保model.py在相同目录或正确路径

warnings.filterwarnings('ignore')

# 特征列表（包含新增的三个微生物特征，共11个特征）
COLUMN_ORDER = [
    '姓名',
    'Periodontal pocket',  # 1. 牙周袋深度
    'CAL',  # 2. 临床附着丧失
    'Missing teeth',  # 3. 缺失牙数
    'Occlusal',  # 4. 咬合状态
    'Looseness',  # 5. 最大牙齿松动度
    'Root bifurcation lesion',  # 6. 最大根分叉病变
    'Gingival condition',  # 7. 牙龈状况
    'inference_code',  # 8. 第8个特征（已替换为inference_code）
    'Porphyromonas_endodontalis',  # 9. 新增特征：牙髓卟啉单胞菌
    'Porphyromonas_gingivalis',  # 10. 新增特征：中间密螺旋体
    'Campylobacter_gracilis'  # 11. 新增特征：纤细弯曲菌
]


def collect_patient_data():
    """收集患者牙周炎临床数据（含11个特征），强化指标验证"""
    patients = []
    while True:
        patient_name = input("请输入患者姓名（输入q退出）：")
        if patient_name.lower() == 'q':
            break

        # 姓名非空验证
        if not patient_name.strip():
            print("错误：患者姓名不能为空，请重新输入。")
            continue

        # 1. 牙周袋深度（正数）
        while True:
            pocket_input = input("请输入患者牙周袋深度（单位：mm，直接输入数字）：")
            try:
                periodontal_pocket = float(pocket_input)
                if periodontal_pocket <= 0:
                    print("错误：牙周袋深度必须为正数，请重新输入。")
                else:
                    break
            except ValueError:
                print("错误：牙周袋深度必须是数字，请重新输入。")

        # 2. CAL（非负数）
        while True:
            cal_input = input("请输入患者临床附着丧失（CAL，单位：mm，直接输入数字）：")
            try:
                cal = float(cal_input)
                if cal < 0:
                    print("错误：临床附着丧失不能为负数，请重新输入。")
                else:
                    break
            except ValueError:
                print("错误：临床附着丧失必须是数字，请重新输入。")

        # 3. 缺失牙数（非负整数）
        while True:
            missing_teeth_input = input("请输入患者缺失牙数（直接输入整数）：")
            try:
                missing_teeth = int(missing_teeth_input)
                if missing_teeth < 0:
                    print("错误：缺失牙数不能为负数，请重新输入。")
                else:
                    break
            except ValueError:
                print("错误：缺失牙数必须是整数，请重新输入。")

        # 4. 咬合状态（0/1/2）
        while True:
            occlusal_input = input(
                "请输入咬合状态（0=无异常，1=缺牙少于4颗且牙齿活动度低，2=缺牙超过4颗且全口牙齿高度松动）：")
            if occlusal_input in ['0', '1', '2']:
                occlusal = int(occlusal_input)
                break
            print("错误：输入无效，请输入0、1或2。")

        # 5. 最大牙齿松动度（0-3）
        while True:
            looseness_input = input("请输入最大牙齿松动度（0=None，1=Grade 1，2=Grade 2，3=Grade 3）：")
            if looseness_input in ['0', '1', '2', '3']:
                looseness = int(looseness_input)
                break
            print("错误：输入无效，请输入0-3之间的数字。")

        # 6. 最大根分叉病变（0-3）
        while True:
            bifurcation_input = input("请输入最大根分叉病变（0=None，1=Grade 1，2=Grade 2，3=Grade 3）：")
            if bifurcation_input in ['0', '1', '2', '3']:
                root_bifurcation = int(bifurcation_input)
                break
            print("错误：输入无效，请输入0-3之间的数字。")

        # 7. 牙龈状况（0-3）
        while True:
            gingival_input = input(
                "请输入牙龈状况（0=None，1=偶尔出血无明显不适，2=频繁出血偶有溢脓伴轻度松动，3=持续出血溢脓伴咀嚼困难）：")
            if gingival_input in ['0', '1', '2', '3']:
                gingival_condition = int(gingival_input)
                break
            print("错误：输入无效，请输入0-3之间的数字。")

        # 8. inference_code特征的输入
        while True:
            print("\ninference_code来源：")
            print("1. 手动输入")
            print("2. 通过CT影像自动推理获取")
            inference_choice = input("请选择inference_code获取方式 (1或2): ")

            if inference_choice == '1':
                inference_code_input = input("请输入inference_code（数字）：")
                try:
                    inference_code = float(inference_code_input)
                    if inference_code < 0:
                        print("错误：inference_code不能为负数，请重新输入。")
                    else:
                        break
                except ValueError:
                    print("错误：inference_code必须是数字，请重新输入。")
            elif inference_choice == '2':
                inference_code = run_ct_inference_pipeline()
                if inference_code is not None:
                    print(f"CT影像推理结果: inference_code = {inference_code}")
                    break
                else:
                    print("CT影像推理失败，请选择其他方式或重试。")
            else:
                print("错误：请输入1或2选择获取方式。")

        # 9. Porphyromonas_endodontalis (牙髓卟啉单胞菌)
        while True:
            pep_input = input("\n请输入牙髓卟啉单胞菌 (Porphyromonas_endodontalis) 数值：")
            try:
                pep = float(pep_input)
                if pep < 0:
                    print("错误：数值不能为负数，请重新输入。")
                else:
                    break
            except ValueError:
                print("错误：必须输入数字，请重新输入。")

        # 10. Treponema_medium (中间密螺旋体)
        while True:
            tm_input = input("请输入中间密螺旋体 (Treponema_medium) 数值：")
            try:
                tm = float(tm_input)
                if tm < 0:
                    print("错误：数值不能为负数，请重新输入。")
                else:
                    break
            except ValueError:
                print("错误：必须输入数字，请重新输入。")

        # 11. Campylobacter_gracilis (纤细弯曲菌)
        while True:
            cg_input = input("请输入纤细弯曲菌 (Campylobacter_gracilis) 数值：")
            try:
                cg = float(cg_input)
                if cg < 0:
                    print("错误：数值不能为负数，请重新输入。")
                else:
                    break
            except ValueError:
                print("错误：必须输入数字，请重新输入。")

        # 按11个特征顺序组织数据
        patients.append({
            '姓名': patient_name,
            'Periodontal pocket': periodontal_pocket,
            'CAL': cal,
            'Missing teeth': missing_teeth,
            'Occlusal': occlusal,
            'Looseness': looseness,
            'Root bifurcation lesion': root_bifurcation,
            'Gingival condition': gingival_condition,
            'inference_code': inference_code,
            'Porphyromonas_endodontalis': pep,
            'Treponema_medium': tm,
            'Campylobacter_gracilis': cg
        })
        print(f"\n患者 {patient_name} 的牙周数据（含11个特征）已记录。")

    return patients


def save_to_excel(patients, file_path="patient_periodontal_data.xlsx"):
    """保存含11个特征的牙周数据到Excel，支持追加"""
    if not patients:
        print("没有数据可保存。")
        return

    df = pd.DataFrame(patients, columns=COLUMN_ORDER)

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_excel(file_path)
            # 为旧数据补全缺失的特征列
            for col in COLUMN_ORDER:
                if col not in existing_df.columns:
                    existing_df[col] = None
            existing_df = existing_df.reindex(columns=COLUMN_ORDER)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(file_path, index=False)
            print(f"数据（含11个特征）已追加到 {file_path}")
        except Exception as e:
            print(f"追加数据时出错: {e}")
    else:
        try:
            df.to_excel(file_path, index=False)
            print(f"新文件已创建: {file_path}（含11个牙周炎临床特征）")
        except Exception as e:
            print(f"创建文件时出错: {e}")


# 定义二分类 Transformer 模型（用于匹配 junqun.pth 权重）
class PeriodontalTransformerClassifier(torch.nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=1, num_classes=2, dropout=0.1):
        """
        完全匹配权重文件的 Transformer 模型结构：
        - input_size=11（11个特征）
        - hidden_size=64（权重文件中实际的隐藏层维度）
        - num_layers=1（权重文件中实际的编码器层数）
        - num_classes=2（二分类输出：0=轻度，1=重度）
        """
        super(PeriodontalTransformerClassifier, self).__init__()

        # 1. 输入投影层：匹配权重文件的 [64, 11] 维度
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),  # [11, 64]
            torch.nn.LayerNorm(hidden_size)  # [64]
        )

        # 2. Transformer 编码器层（1层）
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,  # 64
            nhead=8,  # 多头注意力头数（64/8=8，符合维度要求）
            dim_feedforward=128,  # 前馈网络维度（权重文件中为128）
            dropout=dropout,
            batch_first=True  # 输入格式：[batch, seq_len, feature]
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 输出投影层：根据权重文件调整为正确结构
        # 错误分析表明，output_proj.0 是一个 LayerNorm，而不是 Linear。
        self.output_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),  # output_proj.0, 权重形状为 [64]
            torch.nn.Linear(hidden_size, num_classes)  # output_proj.1, 权重形状为 [2, 64]
        )

    def forward(self, x):
        """前向传播"""
        # 输入投影：[batch, 1, 11] -> [batch, 1, 64]
        x = self.input_proj(x)

        # Transformer 编码：[batch, 1, 64] -> [batch, 1, 64]
        x = self.encoder(x)

        # 取序列最后一个时间步输出：[batch, 1, 64] -> [batch, 64]
        x = x[:, -1, :]

        # 输出投影：[batch, 64] -> [batch, 2]
        logits = self.output_proj(x)
        return logits


def load_and_predict(file_path="patient_periodontal_data.xlsx",
                     model_path=r"E:/D/网站/软件网站模块/junqun.pth",  # 二分类模型权重路径
                     scaler_path="periodontal_scaler.pkl"):
    """加载11个特征的牙周数据，用二分类模型预测"""
    try:
        # 1. 读取数据并验证11个特征
        df = pd.read_excel(file_path)
        print(f"成功加载 {len(df)} 条患者牙周数据（含11个特征校验）")

        # 2. 提取训练模型对应的11个特征
        required_features = [
            'Periodontal pocket', 'CAL', 'Missing teeth', 'Occlusal',
            'Looseness', 'Root bifurcation lesion', 'Gingival condition',
            'inference_code',
            'Porphyromonas_endodontalis', 'Treponema_medium', 'Campylobacter_gracilis'
        ]
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"错误：数据缺少必要特征列: {', '.join(missing_features)}")
            return

        # 3. 处理特征数据
        X = df[required_features].copy()
        for col in required_features:
            X[col] = X[col].fillna(0)
        X = X.astype(float)

        # 4. 特征标准化
        try:
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                if len(scaler.feature_names_in_) != len(required_features):
                    raise ValueError(f"标准化器需11个特征，当前仅{len(scaler.feature_names_in_)}个")
                print(f"成功加载标准化器: {scaler_path}（11个特征适配）")
            else:
                raise FileNotFoundError("标准化器文件不存在，将基于11个特征新建")

            X_scaled = scaler.transform(X)
        except (FileNotFoundError, ValueError) as e:
            print(f"提示：{e}，训练新标准化器")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, scaler_path)
            print(f"新标准化器已保存到: {scaler_path}（11个特征）")
        except Exception as e:
            print(f"标准化过程出错: {e}")
            return

        # 5. 加载二分类模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"二分类模型文件不存在: {model_path}")

        # 实例化模型时使用正确的参数（hidden_size=64, num_layers=1）
        model = PeriodontalTransformerClassifier(
            input_size=11,
            hidden_size=64,  # 与权重文件一致
            num_layers=1,  # 与权重文件一致
            num_classes=2
        )

        # 加载权重时处理不匹配的键
        state_dict = torch.load(model_path, weights_only=True)

        # 调整输出层权重的键名
        new_state_dict = {}
        for key, value in state_dict.items():
            if key == 'output_proj.2.weight':
                new_state_dict['output_proj.1.weight'] = value
            elif key == 'output_proj.2.bias':
                new_state_dict['output_proj.1.bias'] = value
            else:
                new_state_dict[key] = value

        # 加载调整后的权重
        model.load_state_dict(new_state_dict)
        model.eval()
        print(f"成功加载二分类Transformer模型: {model_path}（输入11特征，输出2类）")

        # 6. 预测
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # [batch, 1, 11]

        with torch.no_grad():
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, dim=1)  # 输出0或1
        y_pred = y_pred.numpy()

        # 7. 结果映射（0=轻度，1=重度）
        severity_mapping = {1: "轻度牙周炎", 0: "重度牙周炎"}
        df['牙周炎严重程度（二分类）'] = [severity_mapping.get(pred, "未知") for pred in y_pred]
        df['预测标签（1=轻度,0=重度）'] = y_pred  # 保留数字标签

        # 保存结果
        prediction_file = file_path.replace('.xlsx', '_prediction_binary.xlsx')
        df.to_excel(prediction_file, index=False)
        print(f"\n二分类预测结果已保存到: {prediction_file}")

        # 8. 打印详细结果
        print("\n=== 牙周炎严重程度二分类预测结果 ===")
        print(
            f"{'姓名':<10} {'牙周袋深度(mm)':<15} {'CAL(mm)':<10} {'缺失牙数':<10} {'预测结果'} {'预测标签'}")
        print("-" * 120)

        for _, row in df.iterrows():
            print(
                f"{row['姓名']:<10} {row['Periodontal pocket']:<15.1f} {row['CAL']:<10.1f} "
                f"{row['Missing teeth']:<10d} {row['牙周炎严重程度（二分类）']:<12} {row['预测标签（0=轻度,1=重度）']}")

        # 9. 结果统计
        severity_count = df['牙周炎严重程度（二分类）'].value_counts()
        print("\n=== 二分类预测结果统计 ===")
        for severity, count in severity_count.items():
            percentage = (count / len(df)) * 100
            print(f"{severity}: {count}人（{percentage:.1f}%）")

    except FileNotFoundError as e:
        print(f"文件错误：{e}")
    except Exception as e:
        print(f"预测过程出错：{str(e)}")


class CTInferencePipeline:
    def __init__(self, nii_file_path, model_weight_path, patient_excel_path):
        """初始化CT影像推理流水线"""
        self.nii_path = Path(nii_file_path)
        self.model_weight_path = Path(model_weight_path)
        self.patient_excel_path = Path(patient_excel_path)
        self.model = None
        self.nii_data = None
        self.inference_result = None
        self.class_probabilities = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def load_nii_data(self):
        """读取并预处理nii文件数据"""
        try:
            if not self.nii_path.exists():
                raise FileNotFoundError(f"nii文件不存在: {self.nii_path}")
            if self.nii_path.suffix not in ['.nii', '.nii.gz']:
                raise ValueError(f"文件不是nii格式: {self.nii_path.suffix}")

            nii_img = nib.load(str(self.nii_path))
            self.nii_data = nii_img.get_fdata()  # [H, W, D]
            # 预处理：通道前置+批次维度
            self.nii_data = np.expand_dims(self.nii_data, axis=0)  # [1, H, W, D]
            self.nii_data = np.expand_dims(self.nii_data, axis=0)  # [1, 1, H, W, D]
            self.nii_data = torch.tensor(self.nii_data, dtype=torch.float32)

            print(f"✅ 成功读取nii文件: {self.nii_path.name}")
            print(f"📊 数据形状（batch, channel, H, W, D）: {self.nii_data.shape}")
            return True

        except Exception as e:
            print(f"❌ 读取nii文件失败: {str(e)}")
            return False

    def load_model(self):
        """加载3D ResNet18模型及权重"""
        try:
            if not self.model_weight_path.exists():
                raise FileNotFoundError(f"模型权重文件不存在: {self.model_weight_path}")
            if self.model_weight_path.suffix != '.pth':
                raise ValueError(f"权重文件不是.pth格式: {self.model_weight_path.suffix}")

            self.model = resnet18_3d(num_classes=3, in_channels=1).to(self.device)
            # 处理多GPU权重键名问题
            state_dict = torch.load(str(self.model_weight_path), map_location=self.device)
            model_state_dict = self.model.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k[7:] if k.startswith('module.') else k
                if new_key in model_state_dict:
                    new_state_dict[new_key] = v
                else:
                    print(f"⚠️  跳过不匹配的权重键: {k}")

            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()

            print(f"✅ 成功加载模型权重: {self.model_weight_path.name}")
            print(f"💻 模型运行设备: {self.device}")
            return True

        except Exception as e:
            print(f"❌ 加载模型失败: {str(e)}")
            return False

    def run_inference(self):
        """运行模型推理"""
        try:
            if self.nii_data is None:
                raise ValueError("请先调用load_nii_data()加载数据")
            if self.model is None:
                raise ValueError("请先调用load_model()加载模型")

            with torch.no_grad():
                input_data = self.nii_data.to(self.device)
                logits = self.model(input_data)
                self.class_probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class_code = np.argmax(self.class_probabilities) + 1  # 1/2/3编码
                self.inference_result = pred_class_code

            print(f"\n=== CT影像推理结果 ===")
            print(f"📈 类别概率:")
            print(f"   - 类1 (less_than_1_3): {self.class_probabilities[0]:.4f}")
            print(f"   - 类2 (1_3_to_2_3):   {self.class_probabilities[1]:.4f}")
            print(f"   - 类3 (more_than_2_3): {self.class_probabilities[2]:.4f}")
            print(f"🏆 预测类别编码: {self.inference_result}")
            return True

        except Exception as e:
            print(f"❌ 推理失败: {str(e)}")
            return False

    def get_inference_code(self):
        """获取推理结果"""
        return self.inference_result


def run_ct_inference_pipeline():
    """运行CT影像推理流水线并返回inference_code"""
    # 配置参数
    NII_FILE_PATH = "E:/D/data_set/raw_dataset/test/CT/volume-0.nii"
    MODEL_WEIGHT_PATH = "E:/D/软件网站模块/fenlei.pth"  # CT模型权重路径
    PATIENT_EXCEL_PATH = "E:/D/软件网站模块/patient_periodontal_data.xlsx"  # 患者Excel路径

    print("=== CT影像推理流水线启动 ===")
    pipeline = CTInferencePipeline(
        nii_file_path=NII_FILE_PATH,
        model_weight_path=MODEL_WEIGHT_PATH,
        patient_excel_path=PATIENT_EXCEL_PATH
    )

    # 步骤1：加载nii数据
    if not pipeline.load_nii_data():
        print("❌ CT影像推理终止（数据加载失败）")
        return None

    # 步骤2：加载模型
    if not pipeline.load_model():
        print("❌ CT影像推理终止（模型加载失败）")
        return None

    # 步骤3：运行推理
    if not pipeline.run_inference():
        print("❌ CT影像推理终止（推理失败）")
        return None

    return pipeline.get_inference_code()


if __name__ == "__main__":
    print("=== 患者牙周炎临床数据（11个特征）录入与二分类预测系统 ===")
    while True:
        print("\n请选择操作：")
        print("1. 录入新患者牙周数据（含11个特征）")
        print("2. 使用已有Excel数据进行二分类预测（0=轻度，1=重度）")
        print("3. 仅运行CT影像推理获取inference_code")
        print("4. 退出")
        choice = input("请输入选项 (1-4): ")

        if choice == '1':
            patients_data = collect_patient_data()
            save_to_excel(patients_data)
        elif choice == '2':
            load_and_predict()
        elif choice == '3':
            inference_code = run_ct_inference_pipeline()
            if inference_code is not None:
                print(f"CT影像推理完成，inference_code = {inference_code}")
        elif choice == '4':
            print("系统已退出。")
            break
        else:
            print("无效选项，请输入1-4。")