import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 新增：学习率调度器
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from dataset_lits_val import Val_Dataset
from dataset_lits_train import Train_Dataset
import config
from models import VNet, ResUNet, UNet, U2Net, MagicNet
from utils import logger, weights_init, metrics, common, loss
import torch.optim.lr_scheduler as lr_scheduler
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# Excel日志记录器
class ExcelLogger:
    def __init__(self, file_path):
        """初始化Excel记录器"""
        self.file_path = file_path
        self.columns = [
            'Epoch',
            'Train_Loss', 'Train_dice', 'Train_IoU', 'Train_DSC', 'Train_HD95', 'Train_Valid_Samples',
            'Val_Loss', 'Val_dice', 'Val_IoU', 'Val_DSC', 'Val_HD95', 'Valid_Samples',
            'Timestamp',
            'Learning_Rate'  # 新增：记录当前学习率
        ]

        # 如果文件不存在，创建新文件并写入表头
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_excel(self.file_path, index=False, engine='openpyxl')

    def log_epoch(self, epoch, train_log, val_log, current_lr):  # 新增：传入当前学习率
        """记录一个epoch的指标"""
        # 读取现有数据
        try:
            df = pd.read_excel(self.file_path, engine='openpyxl')
        except:
            df = pd.DataFrame(columns=self.columns)

        # 准备新行数据（新增Learning_Rate字段）
        new_row = {
            'Epoch': epoch,
            'Train_Loss': train_log['Train_Loss'],
            'Train_dice': train_log['Train_dice'],
            'Train_IoU': train_log['Train_IoU'],
            'Train_DSC': train_log['Train_DSC'],
            'Train_HD95': train_log['Train_HD95'],
            'Train_Valid_Samples': train_log['Train_Valid_Samples'],
            'Val_Loss': val_log['Val_Loss'],
            'Val_dice': val_log['Val_dice'],
            'Val_IoU': val_log['Val_IoU'],
            'Val_DSC': val_log['Val_DSC'],
            'Val_HD95': val_log['Val_HD95'],
            'Valid_Samples': val_log['Valid_Samples'],
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Learning_Rate': current_lr  # 记录当前学习率
        }

        # 添加新行
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # 保存回Excel文件
        df.to_excel(self.file_path, index=False, engine='openpyxl')

        # 清理内存
        del df, new_row
        gc.collect()


def visualize_samples(inputs, targets, preds, save_path, title, slice_idx=None):
    """可视化输入图像、目标掩码和预测结果，优化内存占用"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 只处理第一个样本以节省内存
    input_img = inputs[0].cpu().detach().numpy()
    target_mask = targets[0].cpu().detach().numpy()
    pred_mask = preds[0].cpu().detach().numpy()

    # 立即删除大张量引用
    del inputs, targets, preds

    # 处理通道维度
    if input_img.shape[0] == 1:
        input_img = input_img[0]  # [D, H, W]
    if target_mask.shape[0] == 1:
        target_mask = target_mask[0]
    if pred_mask.shape[0] == 1:
        pred_mask = pred_mask[0]

    # 选择要显示的切片
    if slice_idx is None:
        slice_idx = input_img.shape[0] // 2  # 默认中间切片

    # 提取指定切片
    input_slice = input_img[slice_idx]
    target_slice = target_mask[slice_idx]
    pred_slice = pred_mask[slice_idx]

    # 归一化输入图像以便更好地显示
    input_slice = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)

    # 创建图像，使用较小的尺寸
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示输入图像
    axes[0].imshow(input_slice, cmap='gray')
    axes[0].set_title('输入图像')
    axes[0].axis('off')

    # 显示目标掩码
    axes[1].imshow(target_slice, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('目标掩码')
    axes[1].axis('off')

    # 显示预测结果
    axes[2].imshow(pred_slice, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('预测结果')
    axes[2].axis('off')

    # 添加总标题
    plt.suptitle(f'{title} - 切片 {slice_idx}', fontsize=14)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图像，降低DPI减少内存使用
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 清理内存
    del fig, axes, input_img, target_mask, pred_mask
    gc.collect()


def val(model, val_loader, loss_func, voxel_spacing=None, visualize=False, epoch=None, save_dir=None):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage()
    val_metrics = metrics.SegmentationMetrics(2)

    # 用于可视化的样本
    vis_inputs = None
    vis_targets = None
    vis_preds = None

    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            batch_size = data.size(0)

            # 1. 标签处理：转换为双通道独热编码 [B, 2, D, H, W]
            if target.dim() == 4:  # [B, D, H, W]
                target = target.unsqueeze(1)  # [B, 1, D, H, W]
            target_binary = (target > 0).float()  # 二值化 [B, 1, D, H, W]
            # 转换为独热编码：第0通道为背景(1-前景)，第1通道为前景
            target_onehot = torch.cat([1 - target_binary, target_binary], dim=1)  # [B, 2, D, H, W]

            # 确保所有数据都移至设备
            data = data.to(device, non_blocking=True)
            target_onehot = target_onehot.to(device, non_blocking=True)
            target_binary = target_binary.to(device, non_blocking=True)  # 关键：将target_binary移至GPU

            with autocast(device_type='cuda'):
                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]

                # 确保输出和目标维度匹配
                if output.shape[2:] != target_onehot.shape[2:]:
                    output = F.interpolate(output, size=target_onehot.shape[2:], mode='trilinear', align_corners=False)

                # 计算损失（输入和目标均为双通道）
                loss = loss_func(output, target_onehot)

            # 保存第一个批次用于可视化
            if idx == 0 and visualize and epoch is not None and save_dir is not None:
                vis_inputs = data
                vis_targets = target_binary  # 单通道可视化
                vis_preds = torch.sigmoid(output[:, 1:2, ...])  # 只取前景通道

            # 更新指标
            val_loss.update(loss.item(), batch_size)
            pred_probs = torch.sigmoid(output)  # [B, 2, D, H, W]，双通道概率

            # 计算Dice（比较前景通道）- 此时pred_probs和target_binary都在GPU
            val_dice.update(pred_probs[:, 1:2, ...], target_binary)

            # 准备多通道预测结果用于其他指标计算
            output_two_channel = pred_probs  # 已为[B, 2, D, H, W]
            target_metrics = target_binary.squeeze(1).long()  # [B, D, H, W]，单通道类别索引
            target_metrics = target_metrics.to(device, non_blocking=True)  # 确保在GPU

            # 确保预测和目标维度匹配
            if output_two_channel.shape[2:] != target_metrics.shape[1:]:
                output_two_channel = F.interpolate(
                    output_two_channel,
                    size=target_metrics.shape[1:],
                    mode='trilinear',
                    align_corners=False
                )

            # 更新指标计算
            val_metrics.update(output_two_channel, target_metrics, voxel_spacing)

            # 清理变量
            del data, target, target_binary, target_onehot, output, pred_probs, output_two_channel, target_metrics, loss
            if idx % 5 == 0:
                torch.cuda.empty_cache()

    # 生成可视化图像
    if visualize and vis_inputs is not None and epoch is not None and save_dir is not None:
        vis_path = os.path.join(save_dir, 'val_visualization.png')
        visualize_samples(vis_inputs, vis_targets, vis_preds, vis_path, f'验证集可视化 (Epoch {epoch})')
        print(f"验证集可视化图像已保存至: {vis_path}")
        del vis_inputs, vis_targets, vis_preds

    # 整理指标结果
    metrics_results = val_metrics.get_metrics()
    val_log = OrderedDict({
        'Val_Loss': val_loss.avg,
        'Val_dice': val_dice.avg[1],
        'Val_IoU': metrics_results.get('IoU_1', 0),
        'Val_DSC': metrics_results.get('DSC_1', 0),
        'Val_HD95': metrics_results.get('HD95_1', 0),
        'Valid_Samples': metrics_results.get('Valid_Samples', 0),
    })

    # 清理
    del val_loss, val_dice, val_metrics, metrics_results
    torch.cuda.empty_cache()
    gc.collect()

    return val_log


def train(model, train_loader, optimizer, loss_func, alpha=0.4, visualize=False, epoch=None,
          save_dir=None, gradient_accumulation_steps=2):
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage()
    train_metrics = metrics.SegmentationMetrics(2)

    # 用于可视化的样本
    vis_inputs = None
    vis_targets = None
    vis_preds = None

    # 梯度累积计数器
    optimizer.zero_grad(set_to_none=True)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        batch_size = data.size(0)

        # 1. 标签处理：转换为双通道独热编码 [B, 2, D, H, W]
        if target.dim() == 4:  # [B, D, H, W]
            target = target.unsqueeze(1)  # [B, 1, D, H, W]
        target_binary = (target > 0).float()  # 二值化 [B, 1, D, H, W]
        # 独热编码：第0通道背景，第1通道前景
        target_onehot = torch.cat([1 - target_binary, target_binary], dim=1)  # [B, 2, D, H, W]

        # 确保所有张量都移至GPU
        data = data.to(device, non_blocking=True)
        target_onehot = target_onehot.to(device, non_blocking=True)
        target_binary = target_binary.to(device, non_blocking=True)  # 单独移至GPU

        with autocast(device_type='cuda'):
            outputs = model(data)

            # 确保输出和目标维度匹配
            if isinstance(outputs, tuple):
                # 处理多输出模型
                for i, o in enumerate(outputs):
                    if o.shape[2:] != target_onehot.shape[2:]:
                        outputs[i] = F.interpolate(o, size=target_onehot.shape[2:], mode='trilinear', align_corners=False)
                # 计算多输出损失
                loss = sum(loss_func(o, target_onehot) * (alpha if i < len(outputs) - 1 else 1 - alpha)
                           for i, o in enumerate(outputs)) / gradient_accumulation_steps
            else:
                # 处理单输出模型
                if outputs.shape[2:] != target_onehot.shape[2:]:
                    outputs = F.interpolate(outputs, size=target_onehot.shape[2:], mode='trilinear', align_corners=False)
                loss = loss_func(outputs, target_onehot) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 保存可视化样本
        if idx == 0 and visualize and epoch is not None and save_dir is not None:
            vis_inputs = data
            vis_targets = target_binary  # 单通道可视化
            main_output = outputs[-1] if isinstance(outputs, tuple) else outputs
            vis_preds = torch.sigmoid(main_output[:, 1:2, ...])  # 只取前景通道

        # 更新指标
        train_loss.update(loss.item() * gradient_accumulation_steps, batch_size)
        main_output = outputs[-1] if isinstance(outputs, tuple) else outputs
        pred_probs = torch.sigmoid(main_output)  # [B, 2, D, H, W]

        # 计算Dice（此时pred_probs和target_binary都在GPU）
        train_dice.update(pred_probs[:, 1:2, ...], target_binary)

        # 准备多通道预测结果用于其他指标计算
        output_two_channel = pred_probs  # 已为[B, 2, D, H, W]
        target_metrics = target_binary.squeeze(1).long()  # [B, D, H, W]
        target_metrics = target_metrics.to(device, non_blocking=True)  # 确保在GPU

        # 确保预测和目标维度匹配
        if output_two_channel.shape[2:] != target_metrics.shape[1:]:
            output_two_channel = F.interpolate(
                output_two_channel,
                size=target_metrics.shape[1:],
                mode='trilinear',
                align_corners=False
            )

        # 更新指标计算
        train_metrics.update(output_two_channel, target_metrics)

        # 清理变量
        del data, target, target_binary, target_onehot, outputs, pred_probs, output_two_channel, target_metrics, loss, main_output
        if idx % 5 == 0:
            torch.cuda.empty_cache()

    # 生成可视化图像
    if visualize and vis_inputs is not None and epoch is not None and save_dir is not None:
        vis_path = os.path.join(save_dir, 'train_visualization.png')
        visualize_samples(vis_inputs, vis_targets, vis_preds, vis_path, f'训练集可视化 (Epoch {epoch})')
        print(f"训练集可视化图像已保存至: {vis_path}")
        del vis_inputs, vis_targets, vis_preds

    # 整理指标
    metrics_results = train_metrics.get_metrics()
    train_log = OrderedDict({
        'Train_Loss': train_loss.avg,
        'Train_dice': train_dice.avg[1],
        'Train_IoU': metrics_results.get('IoU_1', 0),
        'Train_DSC': metrics_results.get('DSC_1', 0),
        'Train_HD95': metrics_results.get('HD95_1', 0),
        'Train_Valid_Samples': metrics_results.get('Valid_Samples', 0)
    })

    del train_loss, train_dice, train_metrics, metrics_results
    torch.cuda.empty_cache()
    gc.collect()

    return train_log


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):  # 新增scheduler参数
    """从检查点加载模型、优化器、调度器和训练状态"""
    if os.path.exists(checkpoint_path):
        print(f"正在加载检查点: {checkpoint_path}")

        # 导入必要模块以处理安全加载
        import numpy
        from torch.serialization import safe_globals

        # 安全加载上下文管理器
        with safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )

        start_epoch = checkpoint['epoch'] + 1
        best_metrics = checkpoint.get('best_metrics', {
            'dice': 0,
            'iou': 0,
            'dsc': 0,
            'hd95': float('inf')
        })

        # 处理DataParallel的状态字典前缀
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}  # 移除module.前缀

        # 加载模型状态
        model.load_state_dict(state_dict, strict=False)

        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载学习率调度器状态（新增）
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载GradScaler状态
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"成功从 epoch {checkpoint['epoch']} 恢复，继续从 epoch {start_epoch} 训练")

        # 清理内存
        del checkpoint, state_dict
        gc.collect()

        return start_epoch, best_metrics
    else:
        print("未找到检查点，开始新训练")
        return 1, {
            'dice': 0,
            'iou': 0,
            'dsc': 0,
            'hd95': float('inf')
        }


if __name__ == '__main__':
    args = config.args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    save_dir = 'saved_models'
    vis_dir = os.path.join(save_dir, 'visualizations')  # 可视化图像保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 指定要加载的检查点路径
    checkpoint_path = os.path.join(save_dir, 'epoch_6900_checkpoint.pth')

    # 初始化Excel记录器
    excel_logger = ExcelLogger(os.path.join(save_dir, 'training_metrics.xlsx'))

    # 数据加载，优化参数减少内存使用
    train_loader = DataLoader(
        dataset=Train_Dataset(args),
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=1,  # 预加载优化
        persistent_workers=True  # 保持工作进程
    )
    val_loader = DataLoader(
        dataset=Val_Dataset(args),
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    # 模型初始化
    model = VNet.VNet(in_channel=1, out_channel=2).to(device)

    # 如果使用DataParallel，需要包裹模型
    if torch.cuda.device_count() > 1:
        print(f"使用{torch.cuda.device_count()}个GPU进行训练")
        model = nn.DataParallel(model)

    # 损失函数和优化器
    criterion = loss.MixedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)


    class StepLRWithMin(lr_scheduler.StepLR):
        def __init__(self, optimizer, step_size=50, gamma=0.7, last_epoch=-1, min_lr=1e-8):
            self.min_lr = min_lr
            super().__init__(optimizer, step_size, gamma, last_epoch)

        def get_lr(self):
            # 获取StepLR计算的学习率
            lr_list = super().get_lr()
            # 确保学习率不低于最小值
            return [max(lr, self.min_lr) for lr in lr_list]


    # 使用自定义的学习率调度器
    scheduler = StepLRWithMin(
        optimizer,
        step_size=50,  # 每50个epoch调整一次
        gamma=0.7,  # 学习率调整倍数
        last_epoch=-1,  # 最后一个epoch的索引，-1表示从头开始
        min_lr=1e-8  # 学习率下限
    )

    scaler = GradScaler()  # 用于混合精度训练的梯度缩放器

    # 加载检查点（传入scheduler）
    start_epoch, best_metrics = load_checkpoint(
        model, optimizer, scheduler, scaler, checkpoint_path, device
    )

    # 如果是新训练，初始化权重
    if start_epoch == 1:
        model.apply(weights_init.init_model)

    # 训练循环配置
    gradient_accumulation_steps = 4  # 梯度累积步数，可根据内存情况调整
    val_frequency = 1  # 每1个epoch验证一次
    vis_frequency = 10  # 每10个epoch可视化一次

    for epoch in range(start_epoch, args.epochs + 1):
        # 获取当前学习率（用于日志记录）
        current_lr = optimizer.param_groups[0]['lr']

        # 训练
        visualize = (epoch % vis_frequency == 0)
        train_log = train(
            model, train_loader, optimizer, criterion, alpha=0.4,
            visualize=visualize, epoch=epoch, save_dir=vis_dir,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # 验证（每个epoch都执行）
        val_log = val(
            model, val_loader, criterion,
            visualize=visualize, epoch=epoch, save_dir=vis_dir
        )

        # 更新学习率调度器（移除验证指标参数）
        scheduler.step()  # 移除val_log['Val_DSC']参数

        # 记录指标到Excel（包含当前学习率）
        excel_logger.log_epoch(epoch, train_log, val_log, current_lr)

        # 保存检查点(每100个epoch)
        if epoch % 100 == 0 or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 保存调度器状态
                'scaler_state_dict': scaler.state_dict(),
                'best_metrics': best_metrics,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'epoch_{epoch}_checkpoint.pth'))
            print(f"检查点已保存: epoch_{epoch}_checkpoint.pth")

            # 清理检查点相关变量
            del checkpoint
            gc.collect()

        # 更新最佳模型
        val_dsc = val_log['Val_DSC']
        val_iou = val_log['Val_IoU']
        val_dice = val_log['Val_dice']

        # 检查DSC指标是否有效
        is_dsc_valid = not (np.isnan(val_dsc) or np.isinf(val_dsc))

        # 确定当前评估指标（优先使用DSC）
        if is_dsc_valid:
            current_metric = val_dsc
            best_metric = best_metrics['dsc']
        elif not (np.isnan(val_iou) or np.isinf(val_iou)):
            current_metric = val_iou
            best_metric = best_metrics['iou']
            print(f"警告: Epoch {epoch} DSC值无效，使用IoU作为替代指标")
        elif not (np.isnan(val_dice) or np.isinf(val_dice)):
            current_metric = val_dice
            best_metric = best_metrics['dice']
            print(f"警告: Epoch {epoch} DSC和IoU值均无效，使用Dice作为替代指标")
        else:
            print(f"警告: Epoch {epoch} 所有指标均无效，跳过模型保存")
            continue

        # 比较并保存最佳模型
        if current_metric > best_metric:
            # 更新所有最佳指标（包含DSC）
            best_metrics['dsc'] = val_dsc if is_dsc_valid else best_metrics['dsc']
            best_metrics['iou'] = val_iou
            best_metrics['dice'] = val_dice
            best_metrics['hd95'] = val_log['Val_HD95']

            # 准备要保存的模型状态字典
            model_state_dict = model.state_dict()

            # 检查模型权重是否有效
            valid_weights = True
            for k, v in model_state_dict.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"警告: 模型权重 {k} 包含NaN或Inf值，不保存最佳模型")
                    valid_weights = False
                    break
            del k, v  # 清理变量

            if valid_weights:
                # 只保存模型权重
                torch.save(model_state_dict, os.path.join(save_dir, 'best_model_weights.pth'))
                print(f"新最佳模型已保存于epoch {epoch} (基于DSC):")
                print(f"DSC: {val_dsc if is_dsc_valid else 'nan'}")
                print(f"IoU: {val_iou:.4f}")
                print(f"Dice: {val_dice:.4f}")
                print(f"HD95: {val_log['Val_HD95']:.2f}")

            # 清理
            del model_state_dict

        # 打印epoch结果（包含当前学习率）
        log_str = f'Epoch {epoch} | 当前学习率: {current_lr:.8f}\n'  # 新增学习率显示
        log_str += f'Train - Loss: {train_log["Train_Loss"]:.4f} | '
        log_str += f'Dice: {train_log["Train_dice"]:.4f} | '
        log_str += f'IoU: {train_log["Train_IoU"]:.4f} | '
        log_str += f'HD95: {train_log["Train_HD95"]:.2f} | '
        log_str += f'DSC: {train_log["Train_DSC"]:.4f} | '
        log_str += f'Valid%: {train_log["Train_Valid_Samples"]:.2f}\n'
        log_str += f'Val   - Loss: {val_log["Val_Loss"]:.4f} | '
        log_str += f'Dice: {val_log["Val_dice"]:.4f} | '
        log_str += f'IoU: {val_log["Val_IoU"]:.4f} | '
        log_str += f'HD95: {val_log["Val_HD95"]:.2f} | '
        log_str += f'DSC: {val_log["Val_DSC"]:.4f} | '
        log_str += f'Valid%: {val_log["Valid_Samples"]:.2f}\n'
        log_str += f'Best Val Dice: {best_metrics["dice"]:.4f} | '
        log_str += f'Best Val IoU: {best_metrics["iou"]:.4f} | '
        log_str += f'Best Val DSC: {best_metrics["dsc"]:.4f} |'
        log_str += f'Best Val HD95: {best_metrics["hd95"]:.2f}'
        print(log_str)

        # 每个epoch结束时清理内存
        del train_log, val_log, log_str, current_lr
        if epoch % val_frequency == 0:
            del val_dsc, val_iou, val_dice, is_dsc_valid
        torch.cuda.empty_cache()
        gc.collect()