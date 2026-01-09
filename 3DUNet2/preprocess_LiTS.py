import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.widgets as widgets
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]

class LITS_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args, visualize=False, interactive=False):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels
        self.upper = args.upper
        self.lower = args.lower
        self.visualize = visualize
        self.interactive = interactive
        self.valid_rate = 0.4 # 默认验证集比例为20%

        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

        self.colors = {
            0: [0, 0, 0, 0],  # 背景透明
            1: [1, 0, 0, 0.5],  # 肝脏红色半透明
            2: [0, 1, 0, 0.5],  # 肿瘤绿色半透明
        }

        if self.classes == 2:
            self.cmap = ListedColormap([self.colors[0], self.colors[1]])
        else:
            self.cmap = ListedColormap([self.colors[0], self.colors[1], self.colors[2]])

    def fix_data(self):
        if not os.path.exists(self.fixed_path):
            os.makedirs(join(self.fixed_path, 'ct'))
            os.makedirs(join(self.fixed_path, 'label'))

        ct_dir = join(self.raw_root_path, 'ct')
        label_dir = join(self.raw_root_path, 'label')

        if not os.path.exists(ct_dir) or not os.path.exists(label_dir):
            print(f"错误：CT目录 {ct_dir} 或标签目录 {label_dir} 不存在！")
            return

        file_list = os.listdir(ct_dir)
        Numbers = len(file_list)
        print('Total numbers of samples is :', Numbers)

        valid_samples = 0
        invalid_samples = []

        for ct_file, i in zip(file_list, range(Numbers)):
            print(f"==== {ct_file} | {i + 1}/{Numbers} ====")

            ct_path = os.path.join(ct_dir, ct_file)
            seg_file = ct_file.replace('volume', 'segmentation')
            seg_path = os.path.join(label_dir, seg_file)

            if not os.path.exists(ct_path) or not os.path.exists(seg_path):
                print(f"警告：{ct_file} 对应的CT或标签文件缺失，跳过！")
                invalid_samples.append(ct_file)
                continue

            new_ct, new_seg = self.process(ct_path, seg_path, classes=self.classes)
            if new_ct is not None and new_seg is not None:
                seg_array = sitk.GetArrayFromImage(new_seg)
                if np.sum(seg_array > 0) == 0:
                    print(f"警告：{ct_file} 处理后标签无有效区域，跳过保存！")
                    invalid_samples.append(ct_file)
                    continue

                # 保存前强制对齐空间信息
                new_seg.SetDirection(new_ct.GetDirection())
                new_seg.SetOrigin(new_ct.GetOrigin())
                new_seg.SetSpacing(new_ct.GetSpacing())

                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label', seg_file))

                self.print_processed_info(new_ct, new_seg, ct_file)

                # 新增：可选保存可视化结果
                if self.visualize:
                    self.visualize_slices(ct_array=sitk.GetArrayFromImage(new_ct),
                                          seg_array=seg_array,
                                          filename=ct_file,
                                          interactive=self.interactive)

                valid_samples += 1
            else:
                invalid_samples.append(ct_file)

        print(f"\n处理完成：有效样本 {valid_samples}/{Numbers}")
        if invalid_samples:
            print(f"无效样本（{len(invalid_samples)}）：{invalid_samples}")

    def process(self, ct_path, seg_path, classes=None):
        try:
            ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)

            seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            # 严格校验轴序和尺寸
            if ct_array.shape != seg_array.shape:
                print(f"警告：{ct_path} CT与标签尺寸不匹配！CT:{ct_array.shape} 标签:{seg_array.shape}")
                # 尝试自动调整轴序
                if ct_array.shape[1:] == seg_array.shape[:2] and ct_array.shape[0] == seg_array.shape[2]:
                    print("尝试修复轴序：标签转置为(z,y,x)")
                    seg_array = seg_array.transpose(2, 0, 1)
                else:
                    return None, None

            print("Ori shape:", ct_array.shape, seg_array.shape)
            print("原始标签唯一值:", np.unique(seg_array))

            # 新增：将255值转换为1
            seg_array[seg_array == 255] = 1

            if classes == 2:
                seg_array[seg_array > 0] = 1  # 将所有大于0的值设为1

            print("处理后标签唯一值:", np.unique(seg_array))

            # 像素值截断
            ct_array[ct_array > self.upper] = self.upper
            ct_array[ct_array < self.lower] = self.lower

            # 根据裁剪后的数组创建新的 SimpleITK 图像对象并设置空间信息
            new_ct = sitk.GetImageFromArray(ct_array)
            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing(ct.GetSpacing())

            new_seg = sitk.GetImageFromArray(seg_array)
            new_seg.SetDirection(ct.GetDirection())
            new_seg.SetOrigin(ct.GetOrigin())
            new_seg.SetSpacing(ct.GetSpacing())

            return new_ct, new_seg

        except Exception as e:
            print(f"处理 {ct_path} 时出错：{str(e)}")
            return None, None

    def visualize_slices(self, ct_array, seg_array, filename, interactive=False):
        """增强版可视化：支持随机切片、特定切片和交互式浏览"""
        if ct_array.ndim != 3 or seg_array.ndim != 3:
            print(f"警告：{filename} 非3D数据，无法可视化！")
            return

        z_size = ct_array.shape[0]
        if z_size == 0:
            print(f"警告：{filename} 切片数量为0！")
            return

        if interactive:
            # 交互式浏览模式
            self._interactive_viewer(ct_array, seg_array, filename)
        else:
            # 自动模式：随机选择切片
            slice_indices = random.sample(range(z_size), min(3, z_size))
            slice_indices.sort()
            self._static_viewer(ct_array, seg_array, filename, slice_indices)

    def _static_viewer(self, ct_array, seg_array, filename, slice_indices):
        """静态查看器：显示指定切片"""
        n_slices = len(slice_indices)
        fig, axes = plt.subplots(n_slices, 3, figsize=(18, n_slices * 6))
        if n_slices == 1:
            axes = np.array([axes])
        fig.suptitle(f"预处理后数据可视化: {filename}", fontsize=14)

        for i, idx in enumerate(slice_indices):
            # CT切片
            ax0 = axes[i, 0]
            ct_im = ax0.imshow(ct_array[idx], cmap='gray')
            ax0.set_title(f'CT切片 #{idx}')
            ax0.axis('off')

            # 添加CT颜色条
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(ct_im, cax=cax)

            # 标签切片
            ax1 = axes[i, 1]
            label_im = ax1.imshow(seg_array[idx], cmap=self.cmap, vmin=0, vmax=self.classes)
            ax1.set_title(f'标签切片 #{idx}')
            ax1.axis('off')

            # 标签统计
            unique, counts = np.unique(seg_array[idx], return_counts=True)
            stats = "\n".join([f"标签 {int(k)}: {v} 像素" for k, v in zip(unique, counts)])
            ax1.text(0.05, 1.05, stats, transform=ax1.transAxes,
                     verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))

            # 添加标签颜色条
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(label_im, cax=cax, ticks=range(self.classes + 1))
            cbar.ax.set_yticklabels(['背景', '肝脏', '肿瘤'][:self.classes + 1])

            # 叠加显示
            ax2 = axes[i, 2]
            ax2.imshow(ct_array[idx], cmap='gray')
            ax2.imshow(seg_array[idx], cmap=self.cmap, interpolation='none', alpha=0.5, vmin=0, vmax=self.classes)
            ax2.set_title(f'CT与标签叠加 #{idx}')
            ax2.axis('off')

            # 标记特征点辅助对齐校验
            y_feat, x_feat = np.where(ct_array[idx] > 1500)  # 骨组织阈值
            if len(y_feat) > 0 and len(x_feat) > 0:
                y, x = y_feat[0], x_feat[0]
                ax0.scatter(x, y, color='yellow', s=50, marker='+', label='特征点')
                ax2.scatter(x, y, color='yellow', s=50, marker='+', label='特征点')
                ax0.legend()
                ax2.legend()

        # 统一图例
        if self.classes == 2:
            legend_elements = [Patch(facecolor='red', alpha=0.5, label='肝脏')]
        else:
            legend_elements = [Patch(facecolor='red', alpha=0.5, label='肝脏'),
                               Patch(facecolor='green', alpha=0.5, label='肿瘤')]
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()

    def _interactive_viewer(self, ct_array, seg_array, filename):
        """交互式查看器：支持滑动浏览所有切片"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"交互式数据可视化: {filename}", fontsize=14)

        # 初始化切片索引
        current_idx = ct_array.shape[0] // 2  # 默认中间切片

        # 创建三个子图
        # CT切片
        ct_ax = axes[0]
        ct_im = ct_ax.imshow(ct_array[current_idx], cmap='gray')
        ct_ax.set_title(f'CT切片 #{current_idx}')
        ct_ax.axis('off')
        divider = make_axes_locatable(ct_ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ct_im, cax=cax)

        # 标签切片
        label_ax = axes[1]
        label_im = label_ax.imshow(seg_array[current_idx], cmap=self.cmap, vmin=0, vmax=self.classes)
        label_ax.set_title(f'标签切片 #{current_idx}')
        label_ax.axis('off')
        divider = make_axes_locatable(label_ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(label_im, cax=cax, ticks=range(self.classes + 1))
        cbar.ax.set_yticklabels(['背景', '肝脏', '肿瘤'][:self.classes + 1])

        # 叠加显示
        overlay_ax = axes[2]
        overlay_ax.imshow(ct_array[current_idx], cmap='gray')
        overlay_ax.imshow(seg_array[current_idx], cmap=self.cmap, interpolation='none', alpha=0.5, vmin=0,
                          vmax=self.classes)
        overlay_ax.set_title(f'CT与标签叠加 #{current_idx}')
        overlay_ax.axis('off')

        # 添加滑块
        slider_ax = plt.axes([0.2, 0.05, 0.65, 0.03])
        slice_slider = widgets.Slider(slider_ax, '切片', 0, ct_array.shape[0] - 1, valinit=current_idx, valstep=1)

        # 添加按钮：随机选择有标签的切片
        button_ax = plt.axes([0.85, 0.05, 0.1, 0.03])
        random_button = widgets.Button(button_ax, '随机切片')

        # 标签统计文本
        stats_text = fig.text(0.02, 0.05, "", fontsize=10)

        def update(val):
            idx = int(slice_slider.val)

            # 更新CT切片
            ct_im.set_data(ct_array[idx])
            ct_ax.set_title(f'CT切片 #{idx}')

            # 更新标签切片
            label_im.set_data(seg_array[idx])
            label_ax.set_title(f'标签切片 #{idx}')

            # 更新叠加图
            overlay_ax.clear()
            overlay_ax.imshow(ct_array[idx], cmap='gray')
            overlay_ax.imshow(seg_array[idx], cmap=self.cmap, interpolation='none', alpha=0.5, vmin=0,
                              vmax=self.classes)
            overlay_ax.set_title(f'CT与标签叠加 #{idx}')
            overlay_ax.axis('off')

            # 更新标签统计
            unique, counts = np.unique(seg_array[idx], return_counts=True)
            stats = "\n".join([f"标签 {int(k)}: {v} 像素" for k, v in zip(unique, counts)])
            stats_text.set_text(stats)

            fig.canvas.draw_idle()

        def random_slice(event):
            # 找到有标签的切片
            labeled_slices = np.where(np.sum(seg_array, axis=(1, 2)) > 0)[0]
            if len(labeled_slices) > 0:
                random_idx = np.random.choice(labeled_slices)
                slice_slider.set_val(random_idx)

        slice_slider.on_changed(update)
        random_button.on_clicked(random_slice)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    def print_processed_info(self, ct, seg, filename):
        print(f"\n=== 处理后图像信息: {filename} ===")
        print(f"CT 尺寸: {ct.GetSize()}")
        print(f"CT 间距: {ct.GetSpacing()}")
        print(f"标签 尺寸: {seg.GetSize()}")
        print(f"标签 间距: {seg.GetSpacing()}")

        seg_array = sitk.GetArrayFromImage(seg)
        voxel_volume = np.prod(seg.GetSpacing()) / 1000  # 转换为cm³

        liver_volume = np.sum(seg_array > 0) * voxel_volume
        if self.classes == 3:
            tumor_volume = np.sum(seg_array == 2) * voxel_volume
            print(f"肝脏体积: {liver_volume:.2f} cm³")
            print(f"肿瘤体积: {tumor_volume:.2f} cm³")
        else:
            print(f"肝脏体积: {liver_volume:.2f} cm³")

        ct_array = sitk.GetArrayFromImage(ct)
        print(f"CT 强度范围: [{np.min(ct_array):.2f}, {np.max(ct_array):.2f}]")
        print(f"CT 均值/标准差: {np.mean(ct_array):.2f} / {np.std(ct_array):.2f}")
        print("=============================\n")

    def write_train_val_name_list(self, valid_rate=None):
        """
        生成训练集和验证集的文件列表
        :param valid_rate: 验证集比例，如果不指定则使用self.valid_rate
        """
        if valid_rate is not None:
            self.valid_rate = valid_rate

        assert hasattr(self, 'valid_rate'), "valid_rate属性未设置"
        assert 0 < self.valid_rate < 1.0, "valid_rate必须在0和1之间"

        data_name_list = os.listdir(join(self.fixed_path, "ct"))
        data_num = len(data_name_list)
        print('Fixed dataset total numbers of samples is:', data_num)
        random.shuffle(data_name_list)

        train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]
        val_name_list = data_name_list[int(data_num * (1 - self.valid_rate)):]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")
        print(f"数据集分割完成 - 训练集: {len(train_name_list)}个样本, 验证集: {len(val_name_list)}个样本")

    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)
            seg_path = os.path.join(self.fixed_path, 'label', name.replace('volume', 'segmentation'))
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()


if __name__ == '__main__':
    raw_dataset_path = r'D:/project/raw_dataset/data'
    fixed_dataset_path = r'D:/project/raw_dataset/fixed.list'

    args = config.args
    # 启用可视化和交互式查看
    tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path, args, visualize=False, interactive=False)
    tool.fix_data()
    tool.write_train_val_name_list()