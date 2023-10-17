#此功能用于封装简单重复功能的工具集合
# -*- coding: UTF-8 -*-
'''
@Project ：DeepLearning 
@File    ：utile.py
@Author  ：23125342-张旭
@Date    ：2023/8/11 17:46 
'''
from torch.utils.data import  Dataset,random_split,DataLoader
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error as mse_fn, mean_absolute_error as mae_fn
import matplotlib.pyplot as plt

def show_result(train_loss, test_loss, mae, rmse):
    plt.figure(figsize=(20, 5))

    # 绘制训练、测试、验证集的损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label="train_loss")
    plt.plot(test_loss, label="test_loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # 绘制MAE曲线
    plt.subplot(1, 3, 2)
    plt.plot(mae, 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')

    # 绘制RMSE曲线
    plt.subplot(1, 3, 3)
    plt.plot(rmse, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error')

    # 显示图像
    plt.tight_layout()
    plt.show()

def eval(y, pred):
    y = y.cpu().numpy()
    pred = pred.cpu().numpy()
    mse = mse_fn(y, pred)
    rmse = math.sqrt(mse)
    mae = mae_fn(y, pred)
    return [rmse, mae]
