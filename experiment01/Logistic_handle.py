import random
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_size = 50
n_data = torch.ones(data_size, 2)  # 数据的基本形态
x1 = torch.normal(2 * n_data, 1)  # shape=(50, 2)
y1 = torch.zeros(data_size)  # 类型0 shape=(50, 1)
x2 = torch.normal(-2 * n_data, 1)  # shape=(50, 2)
y2 = torch.ones(data_size)  # 类型1 shape=(50, 1)
# 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat是合并数据)
x = torch.cat((x1, x2), 0).type(torch.FloatTensor).to(device)
y = torch.cat((y1, y2), 0).type(torch.FloatTensor).to(device)
test_n_data = torch.ones(data_size, 2)
test_x = torch.cat((torch.normal(2 * test_n_data, 1), torch.normal(-2 * test_n_data, 1)), 0).type(torch.FloatTensor).to(
    device)
test_y = torch.cat((torch.zeros(data_size), torch.ones(data_size)), 0).type(torch.FloatTensor).to(device)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)]).to(device)
        yield features.index_select(0, j), labels.index_select(0, j)


W = torch.tensor(np.random.normal(0, 0.01, (2, 1)), dtype=torch.float32).to(device)
b = torch.zeros(1, dtype=torch.float32).to(device)
W.requires_grad = True
b.requires_grad = True


def sigmod(z):
    return 1 / (1 + torch.exp(-1 * z).to(device))


def logisitic(X, w, b):
    return sigmod(torch.mm(X, w).to(device) + b.to(device))


def cross_loss(y_hat, y):
    delta = 1e-7
    return y.view(y_hat.size()) * torch.log(y_hat + delta).to(device)


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


lr = 0.1
num_epochs = 300
batch_size = 10
net = logisitic
loss = cross_loss
plt.figure(figsize=(10, 5))
losses = []
accuracies = []
for epoch in range(num_epochs):
    for X, Y in data_iter(batch_size, x, y):
        l = -1 * torch.sum(loss(net(X, W, b), Y))
        l.backward()
        sgd([W, b], lr, batch_size)
        W.grad.data.zero_()
        b.grad.data.zero_()

    res = net(test_x, W, b)
    train_l = -1 * loss(res, test_y)
    ress = res.view(-1).tolist()
    ys = y.tolist()
    acc = 0
    for i in range(len(ress)):
        t = 0
        if ress[i] > 0.3:
            t = 1
        if t == ys[i]:
            acc += 1
    accuracies.append(acc / len(y))
    losses.append(train_l.mean().item())
    plt.ion()
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.pause(0.1)
    plt.ioff()
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
