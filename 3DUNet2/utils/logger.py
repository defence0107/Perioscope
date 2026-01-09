import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch,random
import numpy as np
from collections import OrderedDict

class Train_Logger():
    def __init__(self,save_path,save_name):
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update(self, epoch, train_log,val_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)
        # item = dict_round(item,4) # 保留小数点后四位有效数字
        print("\033[0;33mTrain:\033[0m",train_log)
        print("\033[0;33mValid:\033[0m",val_log)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item, index=[0])
        if self.log is not None:
            self.log = self.log._append([tmp], ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' % (self.save_path, self.save_name), index=False)

    def update_tensorboard(self,item):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.save_path)
        epoch = item['epoch']
        for key,value in item.items():
            if key != 'epoch': self.summary.add_scalar(key, value, epoch)


class Test_Logger():
    def __init__(self, save_path, save_name):
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update(self, name, log):
        item = OrderedDict({'img_name': name})

        # 处理 None 的情况
        if log is None:
            item['value'] = None  # 或记录为 NaN 等特殊值
            print("\033[0;33mTest:\033[0m", log)
            self.update_csv(item)
            return

        if isinstance(log, dict):
            item.update(log)
        else:
            # Handle numpy arrays, lists, and scalars
            if isinstance(log, (np.ndarray, list)):
                # Convert to list and store with numbered keys
                log_list = log.tolist() if hasattr(log, 'tolist') else list(log)
                for i, val in enumerate(log_list):
                    item[f'value_{i}'] = val
            else:
                # Handle single values
                try:
                    item['value'] = float(log)
                except (TypeError, ValueError) as e:
                    # 处理无法转换为 float 的情况
                    print(f"Warning: Could not convert {log} to float - {e}")
                    item['value'] = None  # 或其他默认值

        print("\033[0;33mTest:\033[0m", log)
        self.update_csv(item)

    def update_csv(self, item):
        # Convert all values to lists of the same length
        max_length = 1
        for value in item.values():
            if isinstance(value, (list, np.ndarray)):
                max_length = max(max_length, len(value))

        fixed_item = {}
        for key, value in item.items():
            if isinstance(value, (list, np.ndarray)):
                fixed_item[key] = value
            else:
                fixed_item[key] = [value] * max_length

        tmp = pd.DataFrame(fixed_item)

        if self.log is not None:
            self.log = pd.concat([self.log, tmp], ignore_index=True)
        else:
            self.log = tmp

        self.log.to_csv(f'{self.save_path}/{self.save_name}.csv', index=False)

def setpu_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)

def dict_round(dic,num):
    for key,value in dic.items():
        dic[key] = round(value,num)
    return dic