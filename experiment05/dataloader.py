#此文件用于加载数据集
# -*- coding: UTF-8 -*-
'''
@Project ：DeepLearning
@File    ：get_data.py
@Author  ：23125342-张旭
@Date    ：2023/8/11 17:48
'''
from torch.utils.data import Dataset
import numpy as np
import torch


class my_Dataset(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class TrafficDataset:
    def __init__(self, train=True, history_steps=6, future_steps=1):
        # Load and preprocess data
        self.raw_data = None
        if train:
            self.raw_data = np.load(r'E:\毕设文件\DeepLearning\experiment05\data\volume_train.npz')['volume']
        else:
            self.raw_data = np.load(r'E:\毕设文件\DeepLearning\experiment05\data\volume_test.npz')['volume']
        print(self.raw_data.shape)
        # print(self.raw_data)
        # input()
        self.min = self.raw_data.min()
        self.max = self.raw_data.max()
        self.data = (self.raw_data - self.min) / (self.max - self.min)

        self.history_steps = history_steps
        self.future_steps = future_steps

    def denormalize(self, x):
        return x * (self.max - self.min) + self.min

    def construct_set(self):
        x = []
        y = []

        len_train = int(self.data.shape[0])
        train_seqs = self.data[:len_train]

        for i in range(len_train - self.history_steps - self.future_steps + 1):
            x.append(train_seqs[i:i + self.history_steps])
            y.append(train_seqs[i + self.history_steps:i + self.history_steps + self.future_steps])

        data_set = my_Dataset(torch.Tensor(x), torch.Tensor(y))
        return data_set
