import pandas as pd
import os
import joblib
import torch
from sklearn.preprocessing import StandardScaler

# 【核心修正】补充缺失的1个特征（根据训练模型，输入特征数为8，已将"Missing_Feature"替换为"inference_code"）
COLUMN_ORDER = [
    '姓名',
    'Periodontal pocket',  # 1. 牙周袋深度
    'CAL',  # 2. 临床附着丧失
    'Missing teeth',  # 3. 缺失牙数
    'Occlusal',  # 4. 咬合状态
    'Looseness',  # 5. 最大牙齿松动度
    'Root bifurcation lesion',  # 6. 最大根分叉病变
    'Gingival condition',  # 7. 牙龈状况
    'inference_code'  # 8. 第8个特征（已替换为inference_code）
]


def collect_patient_data():
    """收集患者牙周炎临床数据（含inference_code特征），强化指标验证"""
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

        # 8. inference_code特征的输入（对应第8个特征）
        while True:
            inference_code_input = input("请输入inference_code（按训练时要求输入数字）：")
            try:
                inference_code = float(inference_code_input)  # 若为整数，可改为int
                # 根据inference_code实际含义添加范围验证（此处示例非负数验证）
                if inference_code < 0:
                    print("错误：inference_code不能为负数，请重新输入。")
                else:
                    break
            except ValueError:
                print("错误：inference_code必须是数字，请重新输入。")

        # 按8个特征顺序组织数据（使用inference_code作为第8特征）
        patients.append({
            '姓名': patient_name,
            'Periodontal pocket': periodontal_pocket,
            'CAL': cal,
            'Missing teeth': missing_teeth,
            'Occlusal': occlusal,
            'Looseness': looseness,
            'Root bifurcation lesion': root_bifurcation,
            'Gingival condition': gingival_condition,
            'inference_code': inference_code  # 第8个特征：inference_code
        })
        print(f"患者 {patient_name} 的牙周数据（含8个特征）已记录。")

    return patients


def save_to_excel(patients, file_path="patient_periodontal_data.xlsx"):
    """保存含8个特征（含inference_code）的牙周数据到Excel，支持追加"""
    if not patients:
        print("没有数据可保存。")
        return

    df = pd.DataFrame(patients, columns=COLUMN_ORDER)

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_excel(file_path)
            # 为旧数据补全缺失的特征列（含inference_code）
            for col in COLUMN_ORDER:
                if col not in existing_df.columns:
                    existing_df[col] = None
            existing_df = existing_df.reindex(columns=COLUMN_ORDER)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(file_path, index=False)
            print(f"数据（含8个特征，含inference_code）已追加到 {file_path}")
        except Exception as e:
            print(f"追加数据时出错: {e}")
    else:
        try:
            df.to_excel(file_path, index=False)
            print(f"新文件已创建: {file_path}（含8个牙周炎临床特征，含inference_code）")
        except Exception as e:
            print(f"创建文件时出错: {e}")


# 【关键修复】定义与训练模型完全匹配的维度：input_size=8，hidden_size=128
class PeriodontalGRUClassifier(torch.nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=4, num_classes=3):
        """
        100%匹配训练模型的结构：
        - input_size=8（训练模型用8个特征，含inference_code）
        - hidden_size=128（训练模型GRU隐藏层维度）
        - num_layers=4（4层GRU，匹配错误信息的l0-l3）
        """
        super(PeriodontalGRUClassifier, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入格式：[batch_size, seq_len, input_size]
            bidirectional=False  # 从错误信息看为单向GRU，若训练时双向需改为True
        )
        self.bn = torch.nn.BatchNorm1d(hidden_size)  # 维度128，匹配训练模型
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)  # 训练模型fc1为128→128（错误信息：128,128）
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)  # 128→3（输出3类）
        self.relu = torch.nn.ReLU()  # 若训练时用其他激活函数，需替换

    def forward(self, x):
        """前向传播：维度完全匹配训练模型"""
        # x: [batch_size, seq_len, 8] → 输入8个特征（含inference_code）
        gru_out, _ = self.gru(x)  # 输出：[batch_size, seq_len, 128]
        last_step_out = gru_out[:, -1, :]  # 取最后时间步：[batch_size, 128]
        bn_out = self.bn(last_step_out)  # BatchNorm：[batch_size, 128]
        fc1_out = self.relu(self.fc1(bn_out))  # fc1：128→128
        final_out = self.fc2(fc1_out)  # fc2：128→3（输出类别）
        return final_out


def load_and_predict(file_path="patient_periodontal_data.xlsx",
                     model_path="E:/D/RNN2/saved_models/best_model.pth",
                     scaler_path="periodontal_scaler.pkl"):
    """加载8个特征（含inference_code）的牙周数据，用匹配维度的模型预测"""
    try:
        # 1. 读取数据并验证8个特征（含inference_code）
        df = pd.read_excel(file_path)
        print(f"成功加载 {len(df)} 条患者牙周数据（含8个特征校验，含inference_code）")

        # 2. 提取训练模型对应的8个特征（含inference_code，与训练时顺序一致）
        required_features = [
            'Periodontal pocket', 'CAL', 'Missing teeth', 'Occlusal',
            'Looseness', 'Root bifurcation lesion', 'Gingival condition',
            'inference_code'  # 第8个特征：inference_code（已替换完成）
        ]
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"错误：数据缺少必要特征列: {', '.join(missing_features)}")
            return

        # 3. 处理特征数据（填充空值、转为float）
        X = df[required_features].copy()
        # 填充空值（根据特征实际含义调整，此处用0示例）
        for col in required_features:
            X[col] = X[col].fillna(0)
        X = X.astype(float)

        # 4. 特征标准化（确保与训练模型的标准化器维度一致，含inference_code）
        try:
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                if len(scaler.feature_names_in_) != len(required_features):
                    raise ValueError(f"标准化器需8个特征（含inference_code），当前仅{len(scaler.feature_names_in_)}个")
                print(f"成功加载标准化器: {scaler_path}（8个特征适配，含inference_code）")
            else:
                raise FileNotFoundError("标准化器文件不存在，将基于8个特征（含inference_code）新建")

            X_scaled = scaler.transform(X)
        except (FileNotFoundError, ValueError) as e:
            print(f"提示：{e}，训练新标准化器")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, scaler_path)
            print(f"新标准化器已保存到: {scaler_path}（8个特征，含inference_code）")
        except Exception as e:
            print(f"标准化过程出错: {e}")
            return

        # 5. 加载匹配维度的PyTorch模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 初始化与训练模型完全一致的结构（输入8个特征，含inference_code）
        model = PeriodontalGRUClassifier(
            input_size=8,  # 8个特征（含inference_code）
            hidden_size=128,  # 隐藏层维度128
            num_layers=4,  # 4层GRU
            num_classes=3  # 3分类
        )
        # 加载权重（weights_only=True消除警告）
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()  # 评估模式
        print(f"成功加载PyTorch GRU模型: {model_path}（维度完全匹配，含inference_code特征）")

        # 6. 预测（适配GRU输入格式）
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # [batch, 1, 8]

        with torch.no_grad():
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, dim=1)
        y_pred = y_pred.numpy()

        # 7. 结果映射与保存
        severity_mapping = {0: "轻度牙周炎", 1: "中度牙周炎", 2: "重度牙周炎"}
        df['牙周炎严重程度'] = [severity_mapping.get(pred, "未知") for pred in y_pred]

        prediction_file = file_path.replace('.xlsx', '_prediction.xlsx')
        df.to_excel(prediction_file, index=False)
        print(f"\n预测结果已保存到: {prediction_file}")

        # 8. 打印详细结果（展示inference_code特征）
        print("\n=== 牙周炎严重程度预测结果 ===")
        print(
            f"{'姓名':<10} {'牙周袋深度(mm)':<15} {'CAL(mm)':<10} {'缺失牙数':<10} {'inference_code':<12} {'牙龈状况':<20} {'预测结果'}")
        print("-" * 100)

        gingival_text_mapping = {
            0: "无异常",
            1: "偶尔出血无不适",
            2: "频繁出血偶溢脓",
            3: "持续出血伴咀嚼困难"
        }

        for _, row in df.iterrows():
            print(
                f"{row['姓名']:<10} {row['Periodontal pocket']:<15.1f} {row['CAL']:<10.1f} "
                f"{row['Missing teeth']:<10d} {row['inference_code']:<12.1f} "  # 展示inference_code特征值
                f"{gingival_text_mapping.get(row['Gingival condition'], '未知'):<20} "
                f"{row['牙周炎严重程度']}"
            )

        # 9. 结果统计
        severity_count = df['牙周炎严重程度'].value_counts()
        print("\n=== 预测结果统计 ===")
        for severity, count in severity_count.items():
            percentage = (count / len(df)) * 100
            print(f"{severity}: {count}人（{percentage:.1f}%）")

    except FileNotFoundError as e:
        print(f"文件错误：{e}")
    except Exception as e:
        print(f"预测过程出错：{str(e)}")


if __name__ == "__main__":
    print("=== 患者牙周炎临床数据（8个特征，含inference_code）录入与严重程度预测系统 ===")
    while True:
        print("\n请选择操作：")
        print("1. 录入新患者牙周数据（含8个特征，含inference_code）")
        print("2. 使用已有Excel数据预测牙周炎严重程度")
        print("3. 退出")
        choice = input("请输入选项 (1-3): ")

        if choice == '1':
            patients_data = collect_patient_data()
            save_to_excel(patients_data)
        elif choice == '2':
            load_and_predict()
        elif choice == '3':
            print("系统已退出。")
            break
        else:
            print("无效选项，请输入1-3。")