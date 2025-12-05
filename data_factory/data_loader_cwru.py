import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


class CWRULoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # 假设 data_path 是包含 TRAIN.tsv, TEST.tsv, VAL.tsv 的目录
        # 根据 TimesNet 的逻辑，CWRU数据已经是切分好的片段 (Sample, 1+Length)

        train_file = os.path.join(data_path, 'TRAIN.tsv')
        test_file = os.path.join(data_path, 'TEST.tsv')
        val_file = os.path.join(data_path, 'VAL.tsv')

        # 读取数据
        # 第一列是标签，后面是数据
        train_df = pd.read_csv(train_file, sep='\t', header=None)
        test_df = pd.read_csv(test_file, sep='\t', header=None)
        val_df = pd.read_csv(val_file, sep='\t', header=None)

        # 提取特征部分 (去除第一列标签)
        self.train = train_df.iloc[:, 1:].values
        self.test = test_df.iloc[:, 1:].values
        self.val = val_df.iloc[:, 1:].values

        # 提取标签 (用于验证和测试评估)
        self.train_labels = train_df.iloc[:, 0].values
        self.test_labels = test_df.iloc[:, 0].values
        self.val_labels = val_df.iloc[:, 0].values

        # 标准化 (仅使用训练集拟合)
        self.scaler.fit(self.train)
        self.train = self.scaler.transform(self.train)
        self.test = self.scaler.transform(self.test)
        self.val = self.scaler.transform(self.val)

        # 转换为 Channel-last 格式: [N, Length, Channel]
        # CWRU 是单变量振动数据，所以 Channel = 1
        self.train = self.train[:, :, np.newaxis]
        self.test = self.test[:, :, np.newaxis]
        self.val = self.val[:, :, np.newaxis]

        print(f"CWRU {mode} dataset loaded.")
        print("Train shape:", self.train.shape)
        print("Test shape:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == 'val':
            return self.val.shape[0]
        elif self.mode == 'test':
            return self.test.shape[0]
        else:  # thre 模式通常使用测试集
            return self.test.shape[0]

    def __getitem__(self, index):
        # CWRU 数据已经是切分好的，直接按索引取即可，不需要像原代码那样做滑动窗口
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        elif self.mode == 'val':
            return np.float32(self.val[index]), np.float32(self.val_labels[index])
        elif self.mode == 'test':
            return np.float32(self.test[index]), np.float32(self.test_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])


def get_loader_cwru(data_path, batch_size, win_size=1000, mode='train'):
    # dataset 参数在这里不重要，因为这个函数专门处理 CWRU
    dataset = CWRULoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)  # 为了避免Windows/某些环境下的多进程问题，设为0
    return data_loader