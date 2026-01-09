import gc
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.nn.parallel import DataParallel
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import scipy.ndimage as ndimage
import config
from matplotlib.widgets import Slider, Button

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def split_into_patches(volume, patch_size=(64, 128, 128), overlap=0.25):
    """将3D体积分割为重叠的小块（减少单次处理内存）"""
    depth, height, width = volume.shape
    patch_d, patch_h, patch_w = patch_size

    # 计算步长（考虑重叠）
    step_d = int(patch_d * (1 - overlap))
    step_h = int(patch_h * (1 - overlap))
    step_w = int(patch_w * (1 - overlap))

    patches = []
    positions = []  # 记录每个patch的起始坐标

    # 遍历所有可能的patch
    for d in range(0, depth, step_d):
        for h in range(0, height, step_h):
            for w in range(0, width, step_w):
                # 计算patch的结束坐标（避免超出边界）
                d_end = min(d + patch_d, depth)
                h_end = min(h + patch_h, height)
                w_end = min(w + patch_w, width)

                # 提取patch（如果小于patch_size则补零）
                patch = volume[d:d_end, h:h_end, w:w_end]
                if patch.shape != patch_size:
                    pad_d = patch_size[0] - patch.shape[0]
                    pad_h = patch_size[1] - patch.shape[1]
                    pad_w = patch_size[2] - patch.shape[2]
                    patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

                patches.append(patch)
                positions.append((d, d_end, h, h_end, w, w_end))

    return patches, positions, patch_size


def merge_patches(patches, positions, output_shape, patch_size=(64, 128, 128)):
    """将处理后的小块合并为完整3D体积"""
    output = np.zeros(output_shape, dtype=np.float32)
    counts = np.zeros(output_shape, dtype=np.float32)  # 用于重叠区域加权平均

    for i, (d, d_end, h, h_end, w, w_end) in enumerate(positions):
        patch = patches[i]
        # 裁剪patch到实际大小（去除补零部分）
        patch_cropped = patch[:d_end - d, :h_end - h, :w_end - w]
        # 累加patch值并计数
        output[d:d_end, h:h_end, w:w_end] += patch_cropped
        counts[d:d_end, h:h_end, w:w_end] += 1

    # 重叠区域取平均（避免重复计算）
    output = output / np.maximum(counts, 1e-6)
    return output


# 模型加载函数（优化：支持半精度推理）
def load_model(model_path, model_name='VNet', n_labels=2, device='cuda', use_half=False):
    """加载预训练模型，支持半精度推理"""
    # 初始化模型
    if model_name == 'VNet':
        from models import VNet
        model = VNet.VNet(in_channel=1, out_channel=n_labels, training=False).to(device)
    elif model_name == 'ResUNet':
        from models import ResUNet
        model = ResUNet.ResUNet(in_channel=1, out_channel=n_labels, training=False).to(device)
    elif model_name == 'UNet':
        from models import UNet
        model = UNet.UNet(in_channel=1, out_channel=n_labels, training=False).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    # 支持多GPU加载
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    # 加载模型权重
    print(f"正在加载模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location=device)

    # 处理DataParallel的前缀
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # 启用半精度推理
    if use_half and device == 'cuda':
        model.half()  # 模型参数转为float16
        print("已启用半精度推理")

    print("模型加载完成")
    return model


# 结果可视化函数（优化：仅处理当前显示的切片，不保存完整3D数组）
def visualize_results(ct_data, pred_mask, gt_mask=None, show_3d=False):
    """轻量化可视化：仅转换当前显示的切片，不保存完整3D数组"""
    # 创建图形和坐标轴
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # 获取最大切片索引
    if torch.is_tensor(ct_data):
        max_slice = ct_data.shape[0] - 1
    else:
        max_slice = ct_data.shape[0] - 1
    slice_idx = max_slice // 2  # 初始显示中间切片

    # 动态更新切片
    def update(slice_idx):
        slice_idx = int(np.clip(slice_idx, 0, max_slice))

        # 显示CT切片（仅转换当前切片）
        if torch.is_tensor(ct_data):
            ct_slice = ct_data[slice_idx].cpu().numpy()  # 只加载当前切片到CPU
        else:
            ct_slice = ct_data[slice_idx]
        axes[0].clear()
        axes[0].imshow(ct_slice, cmap='gray')
        axes[0].set_title(f'CT切片 (索引: {slice_idx})')
        axes[0].axis('off')

        # 显示预测掩码（仅转换当前切片）
        if torch.is_tensor(pred_mask):
            pred_slice = pred_mask[slice_idx].cpu().numpy()
        else:
            pred_slice = pred_mask[slice_idx]
        axes[1].clear()
        axes[1].imshow(ct_slice, cmap='gray')
        axes[1].imshow(pred_slice, cmap='jet', alpha=0.5)
        axes[1].set_title('预测分割结果')
        axes[1].axis('off')

        # 显示GT掩码（若有）
        if gt_mask is not None:
            if torch.is_tensor(gt_mask):
                gt_slice = gt_mask[slice_idx].cpu().numpy()
            else:
                gt_slice = gt_mask[slice_idx]
            axes[2].clear()
            axes[2].imshow(ct_slice, cmap='gray')
            axes[2].imshow(gt_slice, cmap='jet', alpha=0.5)
            axes[2].set_title('真实分割标签')
            axes[2].axis('off')

        fig.canvas.draw_idle()

    # 初始更新
    update(slice_idx)

    # 添加滑块控件
    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
    slider = Slider(ax_slider, '切片索引', 0, max_slice, valinit=slice_idx, valfmt='%d')
    slider.on_changed(update)

    # 添加前进/后退按钮
    ax_prev = plt.axes([0.1, 0.02, 0.05, 0.03])
    ax_next = plt.axes([0.75, 0.02, 0.05, 0.03])
    btn_prev = Button(ax_prev, '◀')
    btn_next = Button(ax_next, '▶')

    def prev_slice(event):
        update(max(0, slice_idx - 1))

    def next_slice(event):
        update(min(max_slice, slice_idx + 1))

    btn_prev.on_clicked(prev_slice)
    btn_next.on_clicked(next_slice)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

    # 3D可视化优化（下采样减少顶点数量）
    if show_3d:
        fig_3d = plt.figure(figsize=(10, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # 下采样掩码（减少内存和计算量）
        pred_mask_np = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask
        pred_down = ndimage.zoom(pred_mask_np, 0.5, order=0)  # 下采样到50%分辨率

        try:
            verts, faces, _, _ = measure.marching_cubes(pred_down, 0.5)
            verts = verts * 2  # 恢复原始坐标比例
            mesh = Poly3DCollection(verts[faces], alpha=0.5)
            mesh.set_facecolor([0, 1, 0])
            ax_3d.add_collection3d(mesh)
            ax_3d.set_xlim(0, pred_mask_np.shape[2])
            ax_3d.set_ylim(0, pred_mask_np.shape[1])
            ax_3d.set_zlim(0, pred_mask_np.shape[0])
            ax_3d.set_title('预测分割结果3D可视化（下采样）')
            plt.show()
        except ValueError as e:
            print(f"3D可视化错误: {e}")
            print("跳过3D可视化...")


# 保存预测结果为医学图像格式（保持不变）
def save_prediction_as_itk(pred_mask, ref_img_path, output_path, threshold=0.5):
    """将预测结果保存为ITK格式的医学图像"""
    # 加载参考图像获取元数据
    ref_img = sitk.ReadImage(ref_img_path)
    ref_array = sitk.GetArrayFromImage(ref_img)

    # 确保预测掩码与参考图像形状兼容
    if pred_mask.shape != ref_array.shape:
        print(f"警告: 预测掩码形状 {pred_mask.shape} 与参考图像 {ref_array.shape} 不匹配，将调整大小")
        pred_mask = ndimage.zoom(
            pred_mask,
            (ref_array.shape[0] / pred_mask.shape[0],
             ref_array.shape[1] / pred_mask.shape[1],
             ref_array.shape[2] / pred_mask.shape[2]),
            order=0
        )

    # 二值化
    pred_binary = (pred_mask > threshold).astype(np.uint8)

    # 创建ITK图像
    pred_img = sitk.GetImageFromArray(pred_binary)

    # 复制参考图像的元数据
    pred_img.CopyInformation(ref_img)

    # 保存图像
    sitk.WriteImage(pred_img, output_path)
    print(f"预测结果已保存至: {output_path}")


# 模型评估主函数（优化：分块推理+内存清理）
def evaluate_model(model, eval_dataset, output_dir='eval_results', visualize=True, save_itk=True, num_classes=2,
                  use_half=False):
    """分块推理+内存清理：大幅降低峰值内存"""
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device  # 获取模型所在设备

    with torch.no_grad():
        for data_loader, img_name in tqdm(eval_dataset, desc="评估进度"):
            print(f"\n处理图像: {img_name}")

            # 1. 加载原始数据（降低精度）
            ct_np = data_loader.data_np.astype(np.float16)  # 转为float16
            output_shape = ct_np.shape
            gt_mask = data_loader.label_np if hasattr(data_loader, 'label_np') else None

            # 2. 分块处理
            patches, positions, patch_size = split_into_patches(ct_np)
            processed_patches = []

            for patch in tqdm(patches, desc="分块推理", leave=False):
                # 转换为tensor并移动到设备
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                patch_tensor = patch_tensor.to(device)
                if use_half:
                    patch_tensor = patch_tensor.half()  # 半精度输入

                # 推理
                with torch.cuda.amp.autocast():
                    output = model(patch_tensor)
                    if isinstance(output, tuple):
                        output = output[-1]
                    output_softmax = torch.softmax(output, dim=1)[:, 0]  # 取目标通道

                # 保存处理后的patch（释放GPU内存）
                processed_patches.append(output_softmax.squeeze().cpu().numpy())
                del patch_tensor, output, output_softmax  # 及时删除变量
                torch.cuda.empty_cache()  # 清理GPU缓存

            # 3. 合并分块结果
            pred_mask = merge_patches(processed_patches, positions, output_shape, patch_size)

            # 4. 可视化和保存
            if visualize:
                visualize_results(ct_np, pred_mask, gt_mask, show_3d=True)

            if save_itk:
                img_path = data_loader.ct.GetFileName()
                save_prediction_as_itk(pred_mask, img_path, os.path.join(output_dir, f"pred_{img_name}.nii.gz"))
                sitk.WriteImage(data_loader.ct, os.path.join(output_dir, f"ct_{img_name}.nii.gz"))
                if gt_mask is not None:
                    gt_img = sitk.GetImageFromArray(gt_mask)
                    gt_img.CopyInformation(data_loader.ct)
                    sitk.WriteImage(gt_img, os.path.join(output_dir, f"gt_{img_name}.nii.gz"))

            # 5. 强制清理内存
            del ct_np, patches, processed_patches, pred_mask, gt_mask
            gc.collect()  # 强制垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理GPU缓存

            print(f"图像 {img_name} 处理完成（内存已清理）")


if __name__ == "__main__":
    # 获取配置参数
    args = config.args

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模型路径
    model_path = os.path.join('fenge.pth')  # 加载最佳模型权重

    # 评估数据路径
    eval_data_path = args.eval_data_path  # 从配置中获取评估数据路径

    # 1. 加载模型，启用半精度
    model = load_model(
        model_path=model_path,
        model_name='VNet',
        n_labels=2,  # 明确设置为2通道输出
        device=device,
        use_half=True  # 启用半精度推理
    )

    # 2. 加载评估数据
    print("准备评估数据集...")
    from fenge1 import Eval_Datasets  # 根据实际情况调整导入路径

    eval_dataset = Eval_Datasets(eval_data_path, args)

    # 3. 执行评估，启用分块推理和半精度
    evaluate_model(
        model=model,
        eval_dataset=eval_dataset,
        output_dir='eval_results',
        visualize=True,
        save_itk=True,
        num_classes=2,
        use_half=True  # 启用半精度推理
    )

    print("所有评估图像处理完成！")