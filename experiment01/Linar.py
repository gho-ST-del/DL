import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
# from d2l import torch as d2l
#%%
# 导入数据
data = pd.read_csv(r'E:\PythonCode\DeepLearning\experiment01\Datasets\PRSA_data_2010.1.1-2014.12.31.csv', encoding='utf-8')

# 数据预处理
# 去除pm2.5列中的缺失值
data = data[data['pm2.5'].notnull()]
mean_pm25 = data['pm2.5'].mean()
std_pm25 = data['pm2.5'].std()
# 将 "PM2.5" 列中的缺失值（NA）填充为平均值
data['pm2.5'].fillna(mean_pm25, inplace=True)
# 减去平均值
data['pm2.5'] = (data['pm2.5'] - mean_pm25)
# 定义字母类别映射关系
cbwd_mapping = {'NE': 1, 'NW': 2, 'SE': 3, 'cv': 4}
# 将 "cbwd" 列中的字母类别转换为数值
data['cbwd'] = data['cbwd'].map(cbwd_mapping)

# 随机挑选1/5的数据作为测试集
test_set = data.sample(frac=0.2)
# 剩下的数据作为训练集
train_set = data.drop(test_set.index)

train_labels = torch.tensor(train_set.iloc[:, 5].values, dtype=torch.float32)
train_data = torch.tensor(train_set.iloc[:, 2:].values, dtype=torch.float32)
test_labels = torch.tensor(test_set.iloc[:, 5].values, dtype=torch.float32)
test_data = torch.tensor(test_set.iloc[:, 2:].values, dtype=torch.float32)
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
#%%
# dataloader
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 测试dataloader
batch_size = 10
for X, y in data_iter(batch_size, train_data, train_labels):
    print(y)
    break
#%%
# 初始化模型参数
lr = 0.01
theta = np.zeros((train_data.shape[1], 1))
epochs = 100
#%%
# 训练模型
def relu(x):
    return np.maximum(x, 0)
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = torch.tensor(np.random.normal(1, 100, (11, 1)), dtype=torch.float32).to(device)
b = torch.zeros(1, dtype=torch.float32).to(device)
W.requires_grad = True
b.requires_grad = True
def custom_loss(y_hat, y):
    return (y_hat - y) ** 2
def liner(X, w, b):
    return torch.mm(X, w).to(device) + b.to(device)

lr = 0.000000001
num_epochs = 300
batch_size = 10
net = liner
loss = custom_loss
losses = []
accuracies = []
for epoch in range(num_epochs):
    for X, Y in data_iter(batch_size, train_data, train_labels):
        l = -1* torch.sum(loss(liner(X.to(device), W, b), Y.to(device)))
        print(W,l)
        input()
        # print(l)
        l.backward()
        # print(X,Y)
        sgd([W, b], lr, batch_size)
        W.grad.data.zero_()
        b.grad.data.zero_()

    res = liner(test_data.to(device), W, b)
    
    train_l = -1 * loss(res, test_labels.to(device))
    ress = res.view(-1).tolist()
    ys = y.tolist()
    acc = 0

    accuracies.append(acc / len(y))
    losses.append(train_l.mean().item())
    print("epoch %d, loss: %f" % (epoch, train_l.item()))

