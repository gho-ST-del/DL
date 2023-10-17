#此文件用于加载数据集
# -*- coding: UTF-8 -*-
'''
@Project ：DeepLearning 
@File    ：get_data.py
@Author  ：23125342-张旭
@Date    ：2023/8/11 17:48 
'''
from torch.utils.data import  Dataset,random_split,DataLoader
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
    def __init__(self):
        # 选取适当的检测器用作序列数据
        self.raw_data = np.load(r'E:\毕设文件\DeepLearning\data\traffic.npz')['data']
        print(self.raw_data.shape)
        # 数据标准化
        self.min = self.raw_data.min()
        self.max = self.raw_data.max()
        self.data = (self.raw_data - self.min) / (self.max - self.min)

    def denormalize(self, x):
        return x * (self.max - self.min) + self.min

    def construct_set(self, train_por=1, test_por=6, window_size=24, label=0,split=0.6):
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        test_x = []
        test_y = []

        len_train = int(self.data.shape[0] * split)
        train_seqs = self.data[:len_train]
        remain_len = len_train+int((self.data.shape[0] - len_train)/2)


        for i in range(train_seqs.shape[0] - window_size):
            train_x.append(train_seqs[i:i+window_size, train_por, :].squeeze())
            train_y.append(train_seqs[i+window_size, test_por, label].squeeze())

        # 补全构造过程

        val_seqs=self.data[len_train:remain_len]
        for i in range(val_seqs.shape[0] - window_size):
            val_x.append(val_seqs[i:i+window_size, train_por, :].squeeze())
            val_y.append(val_seqs[i+window_size, test_por, label].squeeze())


        test_seqs=self.data[remain_len:]
        for i in range(test_seqs.shape[0] - window_size):
            test_x.append(test_seqs[i:i+window_size, train_por, :].squeeze())
            test_y.append(test_seqs[i+window_size, test_por, label].squeeze())
        train_set = my_Dataset(torch.Tensor(train_x), torch.Tensor(train_y))
        val_set = my_Dataset(torch.Tensor(val_x), torch.Tensor(val_y))
        test_set = my_Dataset(torch.Tensor(test_x), torch.Tensor(test_y))
        return train_set, val_set, test_set