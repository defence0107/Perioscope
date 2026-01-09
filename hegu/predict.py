import nibabel as nib
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
# 导入训练代码中定义的3D ResNet模型（需确保model.py路径正确）
from model import resnet18_3d

warnings.filterwarnings('ignore')


class ModelInferencePipeline:
    def __init__(self, nii_file_path, model_weight_path, inference_excel_path, patient_excel_path):
        """
        初始化模型推理流水线（移除nii_file_name相关处理）
        :param nii_file_path: nii文件路径
        :param model_weight_path: 模型权重文件路径（如results/best_model.pth）
        :param inference_excel_path: 推理结果独立保存路径（原功能保留）
        :param patient_excel_path: 患者数据Excel路径（patient_periodontal_data.xlsx）
        """
        self.nii_path = Path(nii_file_path)
        self.model_weight_path = Path(model_weight_path)
        self.inference_excel_path = Path(inference_excel_path)
        self.patient_excel_path = Path(patient_excel_path)  # 患者Excel路径
        self.model = None
        self.nii_data = None
        self.inference_result = None
        self.class_probabilities = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def load_nii_data(self):
        """读取并预处理nii文件数据（逻辑不变）"""
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
        """加载3D ResNet18模型及权重（逻辑不变）"""
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
        """运行模型推理（逻辑不变）"""
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

            print(f"\n=== 推理结果 ===")
            print(f"📈 类别概率:")
            print(f"   - 类0 (less_than_1_3): {self.class_probabilities[0]:.4f}")
            print(f"   - 类1 (1_3_to_2_3):   {self.class_probabilities[1]:.4f}")
            print(f"   - 类2 (more_than_2_3): {self.class_probabilities[2]:.4f}")
            print(f"🏆 预测类别编码: {self.inference_result}")
            print(f"📝 类别含义: 1=less_than_1_3, 2=1_3_to_2_3, 3=more_than_2_3")
            return True

        except Exception as e:
            print(f"❌ 推理失败: {str(e)}")
            return False

    def write_to_inference_excel(self):
        """写入独立推理结果Excel（原功能保留，仅移除inference_excel中的nii_file_name字段）"""
        try:
            if self.inference_result is None or self.class_probabilities is None:
                raise ValueError("请先调用run_inference()获取推理结果")

            result_data = {
                'inference_code': [self.inference_result],
                'inference_label': [
                    'less_than_1_3' if self.inference_result == 1 else
                    '1_3_to_2_3' if self.inference_result == 2 else
                    'more_than_2_3'
                ],
                'prob_less_than_1_3': [round(self.class_probabilities[0], 4)],
                'prob_1_3_to_2_3': [round(self.class_probabilities[1], 4)],
                'prob_more_than_2_3': [round(self.class_probabilities[2], 4)],
                'processing_time': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            result_df = pd.DataFrame(result_data)

            if self.inference_excel_path.exists():
                existing_df = pd.read_excel(str(self.inference_excel_path))
                combined_df = pd.concat([existing_df, result_df], ignore_index=True)
                combined_df.to_excel(str(self.inference_excel_path), index=False)
                print(f"\n✅ 推理结果已追加到: {self.inference_excel_path.name}")
                print(f"📊 推理Excel总记录数: {len(combined_df)}")
            else:
                result_df.to_excel(str(self.inference_excel_path), index=False)
                print(f"\n✅ 新建推理Excel并写入: {self.inference_excel_path.name}")

            return True

        except Exception as e:
            print(f"❌ 写入推理Excel失败: {str(e)}")
            return False

    def write_to_patient_excel(self):
        """
        优化：仅自动创建缺失的inference_code列，将预测类别编码写入patient_periodontal_data.xlsx
        核心逻辑：移除nii_file_name相关处理，直接基于现有患者数据结构写入推理编码
        """
        try:
            # 前置校验：确保已获取推理结果
            if self.inference_result is None:
                raise ValueError("请先调用run_inference()获取推理结果")
            # 前置校验：确保患者Excel存在
            if not self.patient_excel_path.exists():
                raise FileNotFoundError(f"患者Excel不存在: {self.patient_excel_path}")

            # 1. 读取患者Excel
            patient_df = pd.read_excel(str(self.patient_excel_path))

            # 2. 仅自动创建缺失的inference_code列（移除nii_file_name相关处理）
            if 'inference_code' not in patient_df.columns:
                patient_df.insert(len(patient_df.columns), 'inference_code', None)  # 插入到最后一列
                print(f"✅ 患者Excel缺少'inference_code'列，已自动创建（初始值为空）")

            # 3. 提示用户选择写入行（因移除nii_file_name匹配，改为手动指定行索引）
            print(f"\n📋 患者Excel当前数据行数: {len(patient_df)}")
            print("请指定要写入inference_code的行索引（从0开始，输入'-1'表示追加新行）:")
            while True:
                try:
                    target_idx = int(input("输入行索引: ").strip())
                    if target_idx == -1 or (0 <= target_idx < len(patient_df)):
                        break
                    else:
                        print(f"❌ 无效索引！请输入0到{len(patient_df) - 1}之间的整数，或-1追加新行")
                except ValueError:
                    print("❌ 输入错误！请输入整数类型的行索引")

            # 4. 处理目标行写入
            if target_idx == -1:
                # 追加新行：仅填充inference_code，其他列保持空值
                new_row = pd.DataFrame({col: [None] for col in patient_df.columns})
                new_row['inference_code'] = self.inference_result
                patient_df = pd.concat([patient_df, new_row], ignore_index=True)
                print(f"✅ 已追加新行到患者Excel（inference_code: {self.inference_result}）")
            else:
                # 更新指定行：仅修改inference_code字段
                old_code = patient_df.loc[target_idx, 'inference_code']
                patient_df.loc[target_idx, 'inference_code'] = self.inference_result
                print(f"✅ 已更新行索引{target_idx}的inference_code")
                print(f"   旧值: {old_code} → 新值: {self.inference_result}")

            # 5. 保存更新后的患者Excel
            patient_df.to_excel(str(self.patient_excel_path), index=False)
            print(f"✅ 患者Excel已保存: {self.patient_excel_path.name}")
            print(f"📊 患者Excel当前总行数: {len(patient_df)}")
            print(f"📋 患者Excel列列表: {list(patient_df.columns)}")
            return True

        except Exception as e:
            print(f"❌ 写入患者Excel失败: {str(e)}")
            return False


def main():
    # --------------------------
    # 配置参数（请根据实际路径修改）
    # --------------------------
    NII_FILE_PATH = "E:/D/data_set/raw_dataset/test/CT/volume-0.nii"  # 待推理nii文件
    MODEL_WEIGHT_PATH = "results/best_model.pth"  # 训练最佳权重
    INFERENCE_EXCEL_PATH = "results/inference_results.xlsx"  # 独立推理结果保存路径
    PATIENT_EXCEL_PATH = "E:/D/RNN2/patient_periodontal_data.xlsx"  # 目标患者Excel（需写入的文件）

    # --------------------------
    # 执行推理流水线
    # --------------------------
    print("=== 3D ResNet18 推理流水线启动 ===")
    pipeline = ModelInferencePipeline(
        nii_file_path=NII_FILE_PATH,
        model_weight_path=MODEL_WEIGHT_PATH,
        inference_excel_path=INFERENCE_EXCEL_PATH,
        patient_excel_path=PATIENT_EXCEL_PATH
    )

    # 步骤1：加载nii数据
    if not pipeline.load_nii_data():
        print("❌ 流水线终止（数据加载失败）")
        return

    # 步骤2：加载模型
    if not pipeline.load_model():
        print("❌ 流水线终止（模型加载失败）")
        return

    # 步骤3：运行推理
    if not pipeline.run_inference():
        print("❌ 流水线终止（推理失败）")
        return

    # 步骤4：写入独立推理结果Excel（移除nii_file_name字段）
    if not pipeline.write_to_inference_excel():
        print("⚠️  流水线警告（推理结果写入失败，但继续执行患者Excel写入）")

    # 步骤5：写入患者Excel（仅处理inference_code）
    if not pipeline.write_to_patient_excel():
        print("❌ 流水线终止（患者Excel写入失败）")
        return

    print("\n=== 推理流水线全部完成！===")


if __name__ == "__main__":
    main()
