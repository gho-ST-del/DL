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

def show_result(train_loss, test_loss, val_loss, mae, rmse):
    plt.figure(figsize=(20, 5))

    # 绘制训练、测试、验证集的损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label="train_loss")
    plt.plot(test_loss, label="test_loss")
    plt.plot(val_loss, label="val_loss")
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
def train(denormalize_fn,Opt,lr,loss,Net,train,val,test,epoch_num,device,batch_size,output_model=None,early_stop_len=5):
    rmse, mae = [], []
    max = float('inf')
    t = 0
    train_loss=[]
    val_loss=[]
    test_loss=[]
    start_time = time.time()

    net0 = Net.to(device)
    opt = Opt(net0.parameters(),lr=lr)
    for epoch in range(epoch_num):
        loss_list = []
        for data in train:
            x, y = data
            if x.shape[0] < batch_size:
                continue
            output,hidden = net0(x.to(device))
            if output_model is not None:
                y_hat = output_model(output[:,-1,:].squeeze(-1)).squeeze()
            else:
                y_hat = output[:,-1,:].squeeze(-1)

            # print(y_hat.size(),y.size())
            l = loss(y_hat, y.to(device))

            opt.zero_grad()
            l.backward()
            opt.step()
            loss_list.append(l.item())
        train_loss_mean = np.mean(loss_list)
        train_loss.append(train_loss_mean)


        loss_list = []

        for data in test:
            x, y = data
            if x.shape[0] < batch_size:
                continue
            output, hidden = net0(x.to(device))
            if output_model is not None:
                y_hat = output_model(output[:, -1, :].squeeze(-1)).squeeze()
            else:
                y_hat = output[:, -1, :].squeeze(-1)
            l = loss(y_hat, y.to(device))
            loss_list.append(l.item())

        test_loss_mean = np.mean(loss_list)
        test_loss.append(test_loss_mean)


        loss_list = []
        rmse_list, mae_list = [], []
        for data in val:
            x, y = data
            if x.shape[0] < batch_size:
                continue
            output, hidden = net0(x.to(device))
            if output_model is not None:
                y_hat = output_model(output[:, -1, :].squeeze(-1)).squeeze()
            else:
                y_hat = output[:, -1, :].squeeze(-1)
            l = loss(y_hat, y.to(device))
            loss_list.append(l.item())
            y = denormalize_fn(y)
            y_hat = denormalize_fn(y_hat)
            rmse_a, mae_a = eval(y.detach(), y_hat.detach())
            rmse_list.append(rmse_a)
            mae_list.append(mae_a)

        rmse.append(np.mean(rmse_list))
        mae.append(np.mean(mae_list))
        val_loss_mean = np.mean(loss_list)
        val_loss.append(val_loss_mean)


        print("epoch:", epoch + 1, "\n",
              "train_loss:", train_loss_mean, "test_loss:", test_loss_mean,
              "val_loss:", val_loss_mean,"\n",
              "mae:",np.mean(mae_list),
              "rmse:",np.mean(rmse_list)
              )
        if test_loss_mean < max:
            max = test_loss_mean
            t = 0
        else:
            t += 1
            if t >= early_stop_len:
                print('Early stopping.')
                break
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    return train_loss,val_loss,test_loss,mae,rmse
